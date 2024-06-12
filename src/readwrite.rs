use std::fs;

/// Load a matrix from a file in Matrix Market format.
pub fn mm_read(file_path: &str) -> (Vec<usize>, Vec<usize>, Vec<f64>, usize, usize) {
    let string_content = fs::read_to_string(file_path).expect("Unable to read file");

    let (mut size, mut nonzeros) = (0usize, 0usize);
    let mut at_meta_line = false;
    let mut previous_char = char::default();

    let mut working_slice = "".to_string();
    let mut word: u32 = 1;
    let mut meta_line: u32 = 1;

    for c in string_content.chars() {
        if previous_char == '\n' {
            meta_line += 1;
        }

        at_meta_line = at_meta_line || (previous_char == '\n' && c != '%');

        if at_meta_line {
            if c == ' ' {
                word += 1;
            }

            if c != ' ' && c != '\n' {
                if c != '\r' {
                    working_slice.push(c);
                }
            } else {
                if word == 2 {
                    size = working_slice.parse().ok().unwrap();
                } else if word == 3 {
                    nonzeros = working_slice.parse().ok().unwrap();
                }
                working_slice.clear();
            }

            if c == '\n' {
                break;
            }
        }

        // Close loop
        previous_char = c;
    }

    let (mut rows, mut cols, mut vals): (Vec<usize>, Vec<usize>, Vec<f64>) =
        (vec![0; nonzeros], vec![0; nonzeros], vec![0.0; nonzeros]);

    let mut line_counter: u32 = 1;
    let mut kk: usize;

    word = 1;
    previous_char = char::default();
    working_slice.clear();

    for c in string_content.chars() {
        if previous_char == '\n' {
            line_counter += 1;
            word = 1;
        }

        if line_counter > meta_line {
            if previous_char == ' ' {
                word += 1;
            }

            if c != ' ' && c != '\n' {
                if c != '\r' {
                    working_slice.push(c);
                }
            } else {
                kk = (line_counter - meta_line - 1) as usize;

                if word == 1 {
                    rows[kk] = working_slice.parse::<usize>().ok().unwrap() - 1;
                } else if word == 2 {
                    cols[kk] = working_slice.parse::<usize>().ok().unwrap() - 1;
                } else if word == 3 {
                    vals[kk] = working_slice.parse().ok().unwrap();
                }
                working_slice.clear();
            }
        }

        // Close loop
        previous_char = c;
    }
    return (rows, cols, vals, size, nonzeros);
}

pub fn mm_import(_file_path: &str) -> (Vec<usize>, Vec<usize>, Vec<f64>, usize, usize) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    #[case("./data/nemeth26.csv")]
    #[case("./data/GHS_indef_qpband.csv")]
    #[case("./data/FIDAP_ex4.csv")]
    fn testing_mm_read(#[case] file_path: &str) {
        let (rows, cols, vals, _size, nonzeros) = mm_read(file_path);

        assert!(rows.len() == nonzeros);
        assert!(cols.len() == nonzeros);
        assert!(vals.len() == nonzeros);
    }
}
