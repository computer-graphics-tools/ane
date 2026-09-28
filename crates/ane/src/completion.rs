use std::sync::{Condvar, Mutex};

use crate::Error;

pub type Completion = (Mutex<Option<Result<(), Error>>>, Condvar);
