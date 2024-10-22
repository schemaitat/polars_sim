use num::Float;
use std::{mem::swap, ops::{AddAssign, MulAssign, RangeBounds}};
use num::{Num, PrimInt, Unsigned};

pub trait IndPtrStorage: PrimInt + Unsigned + Default + Send + Sync + AddAssign<Self> {
    fn try_index(&self) -> Option<usize>;

    fn index(self) -> usize;

    fn index_unchecked(self) -> usize;
}

pub trait IndexStorage: PrimInt + Unsigned + Default +  Send + Sync {
}

pub trait DataStorage: Num + Copy + MulAssign<Self> + AddAssign<Self> + PartialOrd + Send + Sync + Into<f32> {
}

#[derive(Debug)]
pub struct CsrMatBase<T, I, J> where 
    I : IndPtrStorage,
    J: IndexStorage,
    T : DataStorage 
{
    pub indptr: Vec<I>,
    pub indices: Vec<J>,
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T, I, J> CsrMatBase<T, I, J> where 
    I : IndPtrStorage,
    J: IndexStorage,
    T : DataStorage
{
    pub fn new(
        indptr: Vec<I>,
        indices: Vec<J>,
        data: Vec<T>,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            indptr,
            indices,
            data,
            rows,
            cols,
        }
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn slice(&self, start: usize, len: usize) -> CsrMatBase<T, I, J>
    {
        let mut indptr = Vec::with_capacity(len + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();

        indptr.push(I::default());

        for i in start..start + len {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            let nnz = end - start;
            indptr.push(*indptr.last().unwrap() + nnz);
            indices.extend_from_slice(&self.indices[start.index()..end.index()]);
            data.extend_from_slice(&self.data[start.index()..end.index()]);
        }

        CsrMatBase::new(indptr, indices, data, len, self.cols)
    }
}

impl<T> CsrMatBase<T>
where
    T: num::Num,
{
    pub fn transpose(&self) -> CsrMatBase<T>
    where
        T: Clone,
    {
        let mut indptr = vec![0; self.cols + 1];
        let mut indices = vec![0; self.nnz()];
        let mut data = vec![T::zero(); self.nnz()];

        for row in 0..self.rows {
            for idx in self.indptr[row]..self.indptr[row + 1] {
                let col = self.indices[idx];
                indptr[col] += 1;
            }
        }

        let mut cumsum = 0;

        for index in indptr.iter_mut() {
            let temp = *index;
            *index = cumsum;
            cumsum += temp;
        }
        indptr[self.cols] = cumsum;

        for row in 0..self.rows {
            for idx in self.indptr[row]..self.indptr[row + 1] {
                let col = self.indices[idx];
                let dest = indptr[col];
                indices[dest] = row;
                data[dest] = self.data[idx].clone();
                indptr[col] += 1;
            }
        }

        let mut last = 0;
        for idx in indptr.iter_mut(){
            swap(&mut last, idx);
        }

        CsrMatBase::new(indptr, indices, data, self.cols, self.rows)
    }
}

impl<T> CsrMatBase<T> {
    pub fn normalize_rows(&mut self)
    where
        T: Float,
    {
        for row in 0..self.rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            let mut sum = T::zero();
            for idx in start..end {
                sum = sum + self.data[idx] * self.data[idx];
            }
            let norm = sum.sqrt();
            for idx in start..end {
                self.data[idx] = self.data[idx] / norm;
            }
        }
    }
}

pub fn topn_from_csr_batches<T: PartialOrd + Clone>(
    batches: Vec<CsrMatBase<T>>,
    ntop: usize,
) -> CsrMatBase<T> {
    // naive implementation
    // iterate ove the rows and collect all possible ntop selections
    // to get batches.len() * ntop elements and then select the topn values
    // from that

    assert_eq!(
        batches.iter().map(|b| b.rows).collect::<Vec<_>>(),
        vec![batches.first().unwrap().rows; batches.len()]
    );

    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    let rows = batches.first().unwrap().rows;

    for i in 0..rows {
        let mut candidates = Vec::new();
        for batch in &batches {
            let start = batch.indptr[i];
            let end = batch.indptr[i + 1];
            for idx in start..end {
                candidates.push((batch.indices[idx], batch.data[idx].clone()));
            }
        }
        // candidates contains now all possible topn values for all batches
        let top_candidates = if ntop < candidates.len() {
            candidates
                .select_nth_unstable_by(ntop, |a, b| b.1.partial_cmp(&a.1).unwrap())
                .0
                .to_vec()
        } else {
            candidates
        };

        let mut nnz = 0;
        for (j, v) in top_candidates {
            indices.push(j);
            data.push(v);
            nnz += 1;
        }
        indptr.push(indptr.last().unwrap() + nnz);
    }

    CsrMatBase::new(indptr, indices, data, rows, batches.first().unwrap().cols)
}
