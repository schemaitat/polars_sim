use num::Float;
use num::{Num, PrimInt, Unsigned};
use std::{
    mem::swap,
    ops::{AddAssign, MulAssign},
};

pub trait IndPtrStorage: PrimInt + Unsigned + Default + Send + Sync + AddAssign<Self> {
    fn try_index(&self) -> Option<usize>;

    fn index(self) -> usize;

    fn index_unchecked(self) -> usize;

    fn from_usize(val: usize) -> Self;
}

pub trait IndexStorage: PrimInt + Unsigned + Default + Send + Sync {
    fn try_index(&self) -> Option<usize>;

    fn index(self) -> usize;

    fn index_unchecked(self) -> usize;

    fn from_usize(val: usize) -> Self;
}

pub trait DataStorage:
    Num + Copy + MulAssign<Self> + AddAssign<Self> + PartialOrd + Send + Sync + Into<f64>
{
}

macro_rules! impl_indices_for_unsigned {
    ($($t:ty),*) => {
        $(
            impl IndPtrStorage for $t {
                fn try_index(&self) -> Option<usize> {
                    Some(*self as usize)
                }

                fn index(self) -> usize {
                    self as usize
                }

                fn index_unchecked(self) -> usize {
                    self as usize
                }

                fn from_usize(val: usize) -> Self {
                    val as Self
                }
            }

            impl IndexStorage for $t {
                fn try_index(&self) -> Option<usize> {
                    Some(*self as usize)
                }

                fn index(self) -> usize {
                    self as usize
                }

                fn index_unchecked(self) -> usize {
                    self as usize
                }

                fn from_usize(val: usize) -> Self {
                    val as Self
                }
            }
        )*
    };
}

macro_rules! impl_data_storage {
    ($($t:ty),*) => {
        $(
            impl DataStorage for $t {}
        )*
    };
}

impl_indices_for_unsigned!(u8, u16, u32, u64, u128, usize);
impl_data_storage!(u16, u32, f32, f64);

#[derive(Debug)]
pub struct CsrMatBase<T, I, J>
where
    I: IndPtrStorage,
    J: IndexStorage,
    T: DataStorage,
{
    pub indptr: Vec<I>,
    pub indices: Vec<J>,
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T, I, J> CsrMatBase<T, I, J>
where
    I: IndPtrStorage,
    J: IndexStorage,
    T: DataStorage,
{
    pub fn new(indptr: Vec<I>, indices: Vec<J>, data: Vec<T>, rows: usize, cols: usize) -> Self {
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

    pub fn slice(&self, start: usize, len: usize) -> CsrMatBase<T, I, J> {
        let mut indptr = Vec::with_capacity(len + 1);
        let mut indices = Vec::with_capacity(self.nnz());
        let mut data = Vec::with_capacity(self.nnz());

        let kk_start = self.indptr[start];
        let kk_end = self.indptr[start + len];

        indptr.extend_from_slice(
            &self.indptr[start..start + len + 1]
                .iter()
                .map(|x| *x - kk_start)
                .collect::<Vec<_>>(),
        );
        indices.extend_from_slice(&self.indices[kk_start.index()..kk_end.index()]);
        data.extend_from_slice(&self.data[kk_start.index()..kk_end.index()]);

        CsrMatBase::new(indptr, indices, data, len, self.cols)
    }
}

impl<T, I, J> CsrMatBase<T, I, J>
where
    T: DataStorage,
    I: IndPtrStorage,
    // need to add this bound to make the transpose method work with usize rows
    J: IndexStorage,
{
    pub fn transpose(&self) -> CsrMatBase<T, I, J> {
        let mut indptr = vec![I::zero(); self.cols + 1];
        let mut indices = vec![J::zero(); self.nnz()];
        let mut data = vec![T::zero(); self.nnz()];

        for row in 0..self.rows {
            for idx in self.indptr[row].index()..self.indptr[row + 1].index() {
                let col = self.indices[idx];
                indptr[col.index()] += I::one();
            }
        }

        let mut cumsum = I::zero();

        for index in indptr.iter_mut() {
            let temp = *index;
            *index = cumsum;
            cumsum += temp;
        }
        indptr[self.cols] = cumsum;

        for row in 0..self.rows {
            for idx in self.indptr[row].index()..self.indptr[row + 1].index() {
                let col = self.indices[idx];
                let dest = indptr[col.index()];
                indices[dest.index()] = J::from_usize(row);
                data[dest.index()] = self.data[idx];
                indptr[col.index()] += I::one();
            }
        }

        let mut last = I::zero();
        for idx in indptr.iter_mut() {
            swap(&mut last, idx);
        }

        CsrMatBase::new(indptr, indices, data, self.cols, self.rows)
    }
}

impl<T, I, J> CsrMatBase<T, I, J>
where
    T: DataStorage,
    I: IndPtrStorage,
    J: IndexStorage,
{
    pub fn normalize_rows(&mut self)
    where
        T: Float,
    {
        for row in 0..self.rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            let mut sum = T::zero();
            for idx in start.index()..end.index() {
                sum += self.data[idx] * self.data[idx];
            }
            let norm = sum.sqrt();
            for idx in start.index()..end.index() {
                self.data[idx] = self.data[idx] / norm;
            }
        }
    }
}

pub fn topn_from_csr_batches<T, I, J>(
    batches: Vec<CsrMatBase<T, I, J>>,
    ntop: usize,
) -> CsrMatBase<T, I, J>
where
    T: DataStorage,
    I: IndPtrStorage,
    J: IndexStorage,
{
    // naive implementation
    // iterate ove the rows and collect all possible ntop selections
    // to get batches.len() * ntop elements and then select the topn values
    // from that

    assert_eq!(
        batches.iter().map(|b| b.rows).collect::<Vec<_>>(),
        vec![batches.first().unwrap().rows; batches.len()]
    );

    let mut indptr = Vec::with_capacity(batches.first().unwrap().rows + 1);
    let mut indices = Vec::with_capacity(batches.first().unwrap().rows * ntop);
    let mut data = Vec::with_capacity(batches.first().unwrap().rows * ntop);

    indptr.push(I::zero());

    let rows = batches.first().unwrap().rows;

    for i in 0..rows {
        let mut candidates = Vec::with_capacity(batches.len() * ntop);
        for batch in &batches {
            let start = batch.indptr[i];
            let end = batch.indptr[i + 1];
            for idx in start.index()..end.index() {
                candidates.push((batch.indices[idx], batch.data[idx]));
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

        let mut nnz = I::zero();
        for (j, v) in top_candidates {
            indices.push(j);
            data.push(v);
            nnz += I::one();
        }
        indptr.push(*indptr.last().unwrap() + nnz);
    }

    CsrMatBase::new(indptr, indices, data, rows, batches.first().unwrap().cols)
}
