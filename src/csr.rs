#[derive(Debug)]
pub struct Csr {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

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
