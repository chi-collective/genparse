mod bool_earley;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use pyo3::prelude::{PyResult, pymodule, PyModule, Bound, PyModuleMethods};
use pyo3::{pyclass, pymethods, FromPyObject};
use crate::Symbol::{Nonterminal, Terminal};
use priority_queue::PriorityQueue;
use bool_earley::EarleyBool;

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
struct Node {
    node: (u32, u32),
    edges: Option<Vec<(u32, u32, u32)>>,
    value: f64,
    cursor: usize,
}

impl Node {
    fn new(node: (u32, u32)) -> Self {
        Self {
            node,
            edges: None,
            value: 0.0,
            cursor: 0,
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
    vocab: HashSet<String>,
    initial_column: Arc<Column>,  // it won't be none after initialization; but necessary since Arc is immutable
    empty_weight: f64, // sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ())
    _chart: HashMap<Box<[String]>, Vec<Arc<Column>>>, // make sure to call ensure_chart before accessing
}

impl Earley {

    fn ensure_chart(&mut self, input: &Box<[String]>) {
        // chart(self, x), but doesnt return the chart itself.
        // otherwise, the chart will have the same lifetime as self, which will trigger
        // a borrow-checker error.
        if ! self._chart.contains_key(input) {
            let new_chart = self.compute_chart(input);
            self._chart.insert(input.clone(), new_chart);
        }
    }

    // todo: Earley also does have a chart method, which is exposed as an external API...
    //       we will think about that later; probably refactor so that chart is never exposed

    fn compute_chart(&mut self, input: &Box<[String]>) -> Vec<Arc<Column>> {
        if input.len() == 0 {
            vec![Arc::clone(&self.initial_column)]
        } else {
            self.ensure_chart(&input[..input.len()-1].into());
            let chart = &self._chart[&input[..input.len()-1]];
            let last_chart = self.next_column(chart, input.last().unwrap());
            let mut new_chart = Vec::new();
            // put all columns in chart into new_chart
            new_chart.extend(chart.iter().cloned());
            new_chart.push(Arc::new(last_chart));
            new_chart
        }
    }

    fn next_column(&self, prev_cols: &Vec<Arc<Column>>, token: &String) -> Column {
        let prev_col = prev_cols.last().unwrap();
        let prev_col_read = prev_col;
        let mut next_col = Column::new(prev_col_read.k + 1);

        let mut queue: PriorityQueue<(u32, u32), i64> = PriorityQueue::new();

        let token_symbol = Terminal(token.to_string()); // todo: inefficient; look into borrow<T>
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

            let col_j = &prev_cols[j as usize];
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

    fn predict_initial_column(&mut self) {
        let mut initial_column = Column::new(0);
        self.predict(&mut initial_column);
        self.initial_column = Arc::new(initial_column);
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

    fn is_terminal(&self, x: &String) -> bool {
        self.vocab.contains(x)
    }

    fn next_token_weights(&self, cols: &Vec<Arc<Column>>) -> HashMap<String, f64> {
        let mut q: HashMap<(u32, u32), f64> = HashMap::new();
        q.insert((0, self.start), 1.0);

        let col = cols.last().unwrap();

        let mut p: HashMap<String, f64> = HashMap::new();
        for y in col.waiting_for.keys() {
            match y {
                Terminal(x) => { if !self.is_terminal(x) { continue; } }
                Nonterminal(_) => continue,
            }
            let x = match y {
                Terminal(x) => { if self.is_terminal(x) { x } else { continue; } }
                Nonterminal(_) => continue,
            };
            let mut total = 0.0;
            for &(i, x, ys) in &col.waiting_for[y] {
                if self.unit_ys[ys as usize] {
                    let node = (i, x);
                    let value = self.next_token_weights_helper(node, cols, &mut q);
                    total += col.i_chart[&(i, x, ys)] * value;
                }
            }
            p.insert(x.clone(), total);
        }

        p
    }

    #[allow(non_snake_case)]
    fn next_token_weights_helper(
        &self, top: (u32, u32), cols: &Vec<Arc<Column>>,
        q: &mut HashMap<(u32, u32), f64>,
    ) -> f64 {
        match q.get(&top) {
            Some(&value) => return value,
            _ => {},
        }

        let mut stack = vec![Node::new(top)];

        while !stack.is_empty() {
            let node = stack.last_mut().unwrap();

            let (j, y) = node.node;

            if node.edges.is_none() {
                let mut edges = Vec::new();
                for x in &cols[j as usize].waiting_for[&Nonterminal(y)] {
                    if self.unit_ys[x.2 as usize] {
                        edges.push(*x);
                    }
                }
                node.edges = Some(edges);
            } else if node.cursor == node.edges.as_ref().unwrap().len() {
                q.insert(node.node, node.value);
                stack.pop();
            } else {
                let arc = node.edges.as_ref().unwrap()[node.cursor];
                let (I, X, _) = arc;
                let neighbor = (I, X);
                let neighbor_value = q.get(&neighbor);
                match neighbor_value {
                    None => {
                        stack.push(Node::new(neighbor));
                    }
                    Some(value) => {
                        node.cursor += 1;
                        node.value += &cols[j as usize].i_chart[&arc] * value;
                    }
                }
            }
        }

        q[&top]
    }

}


#[pymethods]
impl Earley {

    #[new] // this indicated to PyO3 that it's equivalent to the Python __init__
    fn new(
        rhs: HashMap<u32, Vec<RHS>>,
        start: u32,
        order: HashMap<u32, u32>,
        order_max: u32,
        outgoing: HashMap<u32, Vec<u32>>,
        first_ys: Vec<Symbol>,
        rest_ys: Vec<u32>,
        unit_ys: Vec<bool>,
        vocab: HashSet<String>,
        empty_weight: f64,
    ) -> Self {
        // use a dummy value to initialize the struct, and then call the method
        // to actually initialize the column.
        // this is because, once the Arc is initialized, it remains immutable.
        let dummy_column = Arc::new(Column::new(0));
        let mut self_ = Self {
            rhs,
            start,
            order,
            order_max,
            outgoing,
            first_ys,
            rest_ys,
            unit_ys,
            vocab,
            initial_column: dummy_column,
            empty_weight,
            _chart: HashMap::new(),
        };
        // dbg!(&self_);
        // self_.predict(&mut self_.initial_column);
        self_.predict_initial_column();
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
        let c_chart = &cols[boxed_input.len()].c_chart;
        let value = c_chart.get(&(0, self.start));

        match value {
            None => 0.0,
            Some(val) => *val,
        }
    }

    fn clear_cache(&mut self) {
        self._chart.clear();
    }

    fn p_next(&mut self, input: Vec<String>) -> HashMap<String, f64> {
        let boxed_input = input.into_boxed_slice();
        self.ensure_chart(&boxed_input);
        let cols = &self._chart[&boxed_input];
        self.next_token_weights(cols)
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn genpa_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Earley>()?;
    m.add_class::<EarleyBool>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{Earley, RHS};
    use crate::Symbol::{Nonterminal, Terminal};
    use std::collections::{HashMap, HashSet};
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
            Nonterminal(0),
            Nonterminal(2),
            Nonterminal(4),
            Terminal(String::from("c")),
            Nonterminal(5),
            Nonterminal(1),
            Nonterminal(3),
        ];

        let rest_ys = vec![0, 0, 0, 0, 0, 0, 0];

        let unit_ys = vec![false, true, true, true, true, true, true];

        let vocab = HashSet::new();

        let mut earley = Earley::new(
            rhs, 0, order, 6,
            outgoing, first_ys, rest_ys, unit_ys, vocab, 0.0
        );

        // Print to verify
        // println!("{:?}", earley);
        let test_input = vec![String::from("c")];
        let weight = earley.compute_weight(test_input);
        assert_relative_eq!(weight, 0.14285714285714285, epsilon = f64::EPSILON);
    }
}
