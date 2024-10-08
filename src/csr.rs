#[derive(Debug)]
pub struct Csr {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

/// Transposes a CSR matrix.
pub fn transpose_csr(csr: &Csr) -> Csr {
    let mut indptr = vec![0; csr.cols + 1];
    let mut indices = vec![0; csr.data.len()];
    let mut data = vec![0.0; csr.data.len()];

    for row in 0..csr.rows {
        for idx in csr.indptr[row]..csr.indptr[row + 1] {
            let col = csr.indices[idx];
            indptr[col] += 1;
        }
    }

    let mut cumsum = 0;

    for index in indptr.iter_mut() {
        let temp = *index;
        *index = cumsum;
        cumsum += temp;
    }
    indptr[csr.cols] = cumsum;

    for row in 0..csr.rows {
        for idx in csr.indptr[row]..csr.indptr[row + 1] {
            let col = csr.indices[idx];
            let dest = indptr[col];
            indices[dest] = row;
            data[dest] = csr.data[idx];
            indptr[col] += 1;
        }
    }

    let mut last = 0;
    for col in 0..=csr.cols {
        let temp = indptr[col];
        indptr[col] = last;
        last = temp;
    }

    Csr {
        indptr,
        indices,
        data,
        rows: csr.cols,
        cols: csr.rows,
    }
}

pub fn normalize_rows(csr: &mut Csr) {
    for row in 0..csr.rows {
        let start = csr.indptr[row];
        let end = csr.indptr[row + 1];
        let mut sum = 0.0;
        for idx in start..end {
            sum += csr.data[idx] * csr.data[idx];
        }
        let norm = sum.sqrt();
        for idx in start..end {
            csr.data[idx] /= norm;
        }
    }
}

pub fn topn_from_csr_batches(batches: Vec<Csr>, ntop: usize) -> Csr{
    // naive implementation
    // iterate ove the rows and collect all possible ntop selections
    // to get batches.len() * ntop elements and then select the topn values
    // from that


    assert_eq!(batches.iter().map(|b| b.rows).collect::<Vec<_>>(), vec![batches.first().unwrap().rows; batches.len()]);

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

        let mut nnz = 0;
        for (j, v) in top_candidates {
            indices.push(j);
            data.push(v);
            nnz += 1;
        }
        indptr.push(indptr.last().unwrap() + nnz);
    }

    Csr {
        indptr,
        indices,
        data,
        rows,
        cols: batches.first().unwrap().cols,
    }


} 
