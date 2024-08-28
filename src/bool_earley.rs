use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use priority_queue::PriorityQueue;
use pyo3::{pyclass, pymethods, FromPyObject};
use Symbol::{Nonterminal, Terminal};

#[derive(Clone, Debug, PartialEq, Eq, Hash, FromPyObject)]
enum Symbol {
    Terminal(String),
    Nonterminal(u32),
}

type RHS = (bool, u32);


#[derive(Debug)]
struct BoolColumn {
    k: u32,
    i_chart: HashMap<(u32, u32, u32), bool>,
    c_chart: HashMap<(u32, u32), bool>,
    waiting_for: HashMap<Symbol, Vec<(u32, u32, u32)>>,
}

impl BoolColumn {
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
    value: bool,
    cursor: usize,
}

impl Node {
    fn new(node: (u32, u32)) -> Self {
        Self {
            node,
            edges: None,
            value: false,
            cursor: 0,
        }
    }
}


#[derive(Debug)]
#[pyclass]
pub struct EarleyBool {
    rhs: HashMap<u32, Vec<RHS>>,
    start: u32,
    order: HashMap<u32, u32>,
    order_max: u32,
    outgoing: HashMap<u32, Vec<u32>>,
    first_ys: Vec<Symbol>,
    rest_ys: Vec<u32>,
    unit_ys: Vec<bool>,
    vocab: HashSet<String>,
    initial_column: Arc<RwLock<BoolColumn>>,
    empty_weight: bool, // sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ())
    _chart: HashMap<Box<[String]>, Vec<Arc<RwLock<BoolColumn>>>>, // make sure to call ensure_chart before accessing
}

impl EarleyBool {

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

    fn compute_chart(&mut self, input: &Box<[String]>) -> Vec<Arc<RwLock<BoolColumn>>> {
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

    fn next_column(&self, prev_cols: &Vec<Arc<RwLock<BoolColumn>>>, token: &String) -> BoolColumn {
        let prev_col = prev_cols.last().unwrap();
        let prev_col_read = prev_col.read().unwrap();
        let mut next_col = BoolColumn::new(prev_col_read.k + 1);

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

            let col_j = &prev_cols[j as usize].read().unwrap();
            let val = next_col.c_chart[&jy];
            if !col_j.waiting_for.contains_key(&Nonterminal(y)) { continue; }
            for customer in col_j.waiting_for[&Nonterminal(y)].iter() {
                let (i, x, ys) = customer;
                self.update_column(
                    &mut next_col, Some(&mut queue), *i, *x,
                    self.rest_ys[*ys as usize],
                    col_j.i_chart[customer] & val,
                );
            }
        }

        self.predict(&mut next_col);

        next_col
    }

    fn predict(&self, col: &mut BoolColumn) {
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

    fn update_column(&self, col: &mut BoolColumn, queue: Option<&mut PriorityQueue<(u32, u32), i64>>, i: u32, x: u32, ys: u32, value: bool) {
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
                col.c_chart.insert(item, was.unwrap() | value);
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
                col.i_chart.insert(item, was.unwrap() | value);
            }
        }
    }

    fn is_terminal(&self, x: &String) -> bool {
        self.vocab.contains(x)
    }

    fn next_token_weights(&self, cols: &Vec<Arc<RwLock<BoolColumn>>>) -> HashMap<String, bool> {
        let mut q: HashMap<(u32, u32), bool> = HashMap::new();
        q.insert((0, self.start), true);

        let col = cols.last().unwrap().read().unwrap();

        let mut p: HashMap<String, bool> = HashMap::new();
        for y in col.waiting_for.keys() {
            match y {
                Terminal(x) => { if !self.is_terminal(x) { continue; } }
                Nonterminal(_) => continue,
            }
            let x = match y {
                Terminal(x) => { if self.is_terminal(x) { x } else { continue; } }
                Nonterminal(_) => continue,
            };
            let mut total = false;
            for &(i, x, ys) in &col.waiting_for[y] {
                if self.unit_ys[ys as usize] {
                    let node = (i, x);
                    let value = self.next_token_weights_helper(node, cols, &mut q);   // todo
                    total |= col.i_chart[&(i, x, ys)] & value;
                }
            }
            p.insert(x.clone(), total);
        }

        p
    }

    #[allow(non_snake_case)]
    fn next_token_weights_helper(
        &self, top: (u32, u32), cols: &Vec<Arc<RwLock<BoolColumn>>>,
        q: &mut HashMap<(u32, u32), bool>,
    ) -> bool {
        match q.get(&top) {
            Some(&value) => return value,
            _ => {},
        }

        let mut stack = vec![Node::new(top)];

        while !stack.is_empty() {
            let node = stack.last_mut().unwrap();

            let (j, y) = node.node;

            if node.edges.is_none() {
                // todo
                let mut edges = Vec::new();
                for x in &cols[j as usize].read().unwrap().waiting_for[&Nonterminal(y)] {
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
                        node.value |= &cols[j as usize].read().unwrap().i_chart[&arc] & value;
                    }
                }
            }
        }

        q[&top]
    }

}


#[pymethods]
impl EarleyBool {

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
        vocab: HashSet<String>,
        empty_weight: bool,
    ) -> Self {
        let initial_column = Arc::new(RwLock::new(BoolColumn::new(0)));
        let self_ = Self {
            rhs,
            start,
            order,
            order_max,
            outgoing,
            first_ys,
            rest_ys,
            unit_ys,
            vocab,
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

    fn compute_weight(&mut self, input: Vec<String>) -> bool { // __call__(self, x)
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
            None => false,
            Some(val) => *val,
        }
    }

    fn clear_cache(&mut self) {
        self._chart.clear();
    }

    fn p_next(&mut self, input: Vec<String>) -> HashMap<String, bool> {
        let boxed_input = input.into_boxed_slice();
        self.ensure_chart(&boxed_input);
        let cols = &self._chart[&boxed_input];
        self.next_token_weights(cols)
    }
}


#[cfg(test)]
mod tests {

}
