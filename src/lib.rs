use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use pyo3::prelude::{PyResult, pymodule, PyModule, Bound, PyModuleMethods};
use pyo3::{pyclass, pymethods, FromPyObject};
use crate::Symbol::{Nonterminal, Terminal};
use priority_queue::PriorityQueue;

#[derive(Clone, Debug, PartialEq, Eq, Hash, FromPyObject)]
enum Symbol {
    Terminal(String),
    Nonterminal(u32),
}

type RHS = (f64, u32);


#[derive(Debug)]
struct Column {
    k: u32,
    i_chart: HashMap<(u32, u32, u32), f64>,
    c_chart: HashMap<(u32, u32), f64>,
    waiting_for: HashMap<Symbol, Vec<(u32, u32, u32)>>,
}

impl Column {
    fn new(k: u32) -> Self {
        Self {
            k,
            i_chart: HashMap::new(),
            c_chart: HashMap::new(),
            waiting_for: HashMap::new(),
        }
    }
}


#[derive(Debug)]
#[pyclass]
struct Earley {
    rhs: HashMap<u32, Vec<RHS>>,
    start: u32,
    order: HashMap<u32, u32>,
    order_max: u32,
    outgoing: HashMap<u32, Vec<u32>>,
    first_ys: Vec<Symbol>,
    rest_ys: Vec<u32>,
    unit_ys: Vec<bool>,
    initial_column: Arc<RwLock<Column>>,
    empty_weight: f64, // sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ())
    _chart: HashMap<Box<[String]>, Vec<Arc<RwLock<Column>>>>,
}

impl Earley {

    fn ensure_chart(&mut self, input: &Box<[String]>) {
        // chart(self, x), but doesnt return the chart itself.
        // otherwise, the chart will have the same lifetime as self, which will trigger
        // a borrow-checker error.
        // let boxed_input: Box<[String]> = input.into();
        let c = self._chart.get(input);
        if c.is_none() {
            let new_chart = self.compute_chart(input);
            self._chart.insert(input.clone(), new_chart);
        }
    }

    // todo: Earley also does have a chart method, which is exposed as an external API...
    //       we will think about that later; probably refactor so that chart is never exposed

    fn compute_chart(&mut self, input: &Box<[String]>) -> Vec<Arc<RwLock<Column>>> {
        if input.len() == 0 {
            vec![Arc::clone(&self.initial_column)]
        } else {
            self.ensure_chart(&input[..input.len()-1].into());
            let chart = &self._chart[&input[..input.len()-1]];
            let last_chart = self.next_column(chart, input.last().unwrap());
            let mut new_chart = Vec::new();
            // put all columns in chart into new_chart
            new_chart.extend(chart.iter().cloned());
            new_chart.push(Arc::new(RwLock::new(last_chart)));
            new_chart
        }
    }

    fn next_column(&self, prev_cols: &Vec<Arc<RwLock<Column>>>, token: &String) -> Column {
        let prev_col = prev_cols.last().unwrap();
        let prev_col_read = prev_col.read().unwrap();
        let mut next_col = Column::new(prev_col_read.k + 1);

        let mut queue: PriorityQueue<(u32, u32), i64> = PriorityQueue::new();

        let token_symbol = Terminal(token.to_string());
        if prev_col_read.waiting_for.contains_key(&token_symbol) {
            for item in prev_col_read.waiting_for[&token_symbol].iter() {
                let (i, x, ys) = item;
                self.update_column(
                    &mut next_col, Some(&mut queue), *i, *x,
                    self.rest_ys[*ys as usize],
                    prev_col_read.i_chart[item],
                )
            }
        }

        while let Some((jy, _)) = queue.pop() {
            let (j, y) = jy;

            let col_j = &prev_cols[j as usize].read().unwrap();
            let val = next_col.c_chart[&jy];
            if !col_j.waiting_for.contains_key(&Nonterminal(y)) { continue; }
            for customer in col_j.waiting_for[&Nonterminal(y)].iter() {
                let (i, x, ys) = customer;
                self.update_column(
                    &mut next_col, Some(&mut queue), *i, *x,
                    self.rest_ys[*ys as usize],
                    col_j.i_chart[customer] * val,
                );
            }
        }

        self.predict(&mut next_col);

        next_col
    }

    fn predict(&self, col: &mut Column) {
        let k = col.k;

        let mut agenda = match k {
            0 => vec![Nonterminal(self.start)],
            _ => col.waiting_for.keys().cloned().collect(),
        };

        let mut reachable: HashSet<Symbol> = agenda.iter().cloned().collect();

        while let Some(x) = agenda.pop() {
            if let Nonterminal(x) = x {
                if !self.outgoing.contains_key(&x) { continue; }
                for y in &self.outgoing[&x] {
                    if !reachable.contains(&Nonterminal(*y)) { // todo: impl u32 borrow trait for symbol
                        reachable.insert(Nonterminal(*y));
                        agenda.push(Nonterminal(*y));
                    }
                }
            }
        }

        // dbg!(&reachable);

        for x in &reachable {
            let x = match x {
                Nonterminal(x) => x,
                _ => continue,
            };
            let ys = match self.rhs.get(&x) {
                Some(ys) => ys,
                None => continue,
            };
            for rhs in ys {
                let (w, ys) = rhs;
                self.update_column(col, None, k, *x, *ys, *w);
            }
        }
    }

    fn update_column(&self, col: &mut Column, queue: Option<&mut PriorityQueue<(u32, u32), i64>>, i: u32, x: u32, ys: u32, value: f64) {
        if ys == 0 {
            let item = (i, x);
            let was = col.c_chart.get(&item);
            if was.is_none() {
                let queue = queue.unwrap();
                let new_priority = -((col.k as i64 - i as i64) * (self.order_max as i64) + (self.order[&x] as i64));

                if queue.get(&item).is_none() {
                    queue.push(item, new_priority);
                } else {
                    queue.change_priority(&item, new_priority);
                }

                col.c_chart.insert(item, value);
            } else {
                col.c_chart.insert(item, was.unwrap() + value);
            }
        } else {
            let item = (i, x, ys);
            let was = col.i_chart.get(&item);
            if was.is_none() {
                col.waiting_for.entry(self.first_ys[ys as usize].clone())
                    .or_insert_with(Vec::new)
                    .push(item);
                col.i_chart.insert(item, value);
            } else {
                col.i_chart.insert(item, was.unwrap() + value);
            }
        }
    }
}


#[pymethods]
impl Earley {

    #[new]
    fn new(
        rhs: HashMap<u32, Vec<RHS>>,
        start: u32,
        order: HashMap<u32, u32>,
        order_max: u32,
        outgoing: HashMap<u32, Vec<u32>>,
        first_ys: Vec<Symbol>,
        rest_ys: Vec<u32>,
        unit_ys: Vec<bool>,
        empty_weight: f64,
    ) -> Self {
        let initial_column = Arc::new(RwLock::new(Column::new(0)));
        let self_ = Self {
            rhs,
            start,
            order,
            order_max,
            outgoing,
            first_ys,
            rest_ys,
            unit_ys,
            initial_column,
            empty_weight,
            _chart: HashMap::new(),
        };
        // dbg!(&self_);
        self_.predict(&mut self_.initial_column.write().unwrap());
        // println!("---------------");
        // dbg!(&self_);
        self_
    }

    fn compute_weight(&mut self, input: Vec<String>) -> f64 { // __call__(self, x)
        if input.len() == 0 {
            return self.empty_weight;
        }

        self._chart.insert(Box::new([]), vec![Arc::clone(&self.initial_column)]);

        let boxed_input = input.into_boxed_slice();
        self.ensure_chart(&boxed_input);
        let cols = self._chart.get(&boxed_input).unwrap();
        let c_chart = &cols[boxed_input.len()].read().unwrap().c_chart;
        let value = c_chart.get(&(0, self.start));

        match value {
            None => 0.0,
            Some(val) => *val,
        }
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn genpa_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Earley>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{Symbol, Earley, RHS};
    use std::collections::{HashMap};
    use approx::assert_relative_eq;

    #[test]
    fn test_simple() {
        let rhs: HashMap<u32, Vec<RHS>> = [
            (3, vec![(0.2857142857142857, 2)]),
            (0, vec![(0.5, 5), (0.5, 6)]),
            (1, vec![(0.2857142857142857, 1)]),
            (2, vec![(1.0, 4)]),
            (4, vec![(1.0, 4)]),
            (5, vec![(0.5, 3)]),
        ]
            .iter()
            .cloned()
            .collect();

        let order: HashMap<u32, u32> = [
            (2, 1),
            (0, 5),
            (5, 0),
            (1, 2),
            (4, 3),
            (3, 4),
        ]
            .iter()
            .cloned()
            .collect();

        let outgoing: HashMap<u32, Vec<u32>> = [
            (1, vec![2]),
            (4, vec![5]),
            (0, vec![1, 3]),
            (2, vec![5]),
            (3, vec![4]),
        ]
            .iter()
            .cloned()
            .collect();

        let first_ys = vec![
            Symbol::Nonterminal(0),
            Symbol::Nonterminal(2),
            Symbol::Nonterminal(4),
            Symbol::Terminal(String::from("c")),
            Symbol::Nonterminal(5),
            Symbol::Nonterminal(1),
            Symbol::Nonterminal(3),
        ];

        let rest_ys = vec![0, 0, 0, 0, 0, 0, 0];

        let unit_ys = vec![false, true, true, true, true, true, true];

        let mut earley = Earley::new(
            rhs, 0, order, 6,
            outgoing, first_ys, rest_ys, unit_ys, 0.0
        );

        // Print to verify
        // println!("{:?}", earley);
        let test_input = vec![String::from("c")];
        let weight = earley.compute_weight(test_input);
        assert_relative_eq!(weight, 0.14285714285714285, epsilon = f64::EPSILON);
    }
}
