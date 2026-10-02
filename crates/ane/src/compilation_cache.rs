use crate::{Error, Executable, NSQualityOfService, Program, require};
use std::{collections::VecDeque, sync::Arc};

pub struct CompilationCache {
    entries: VecDeque<(Program, NSQualityOfService, Executable)>,
    capacity: usize,
    byte_limit: usize,
    bytes: usize,
    hits: usize,
}

impl CompilationCache {
    pub fn new(capacity: usize, byte_limit: usize) -> Result<Self, Error> {
        require(capacity > 0 && byte_limit > 0, Error::CacheLimits)?;
        Ok(Self {
            entries: VecDeque::new(),
            capacity,
            byte_limit,
            bytes: 0,
            hits: 0,
        })
    }
    pub fn compile(
        &mut self,
        program: Program,
        qos: NSQualityOfService,
    ) -> Result<Arc<Executable>, Error> {
        if let Some(index) = self
            .entries
            .iter()
            .position(|(p, q, _)| *q == qos && p == &program)
        {
            let entry = self.entries.remove(index).unwrap();
            let executable = entry.2.clone();
            self.entries.push_back(entry);
            self.hits += 1;
            return Ok(Arc::new(executable));
        }
        let bytes = program.source_bytes();
        require(
            bytes <= self.byte_limit,
            Error::CacheOverflow {
                bytes,
                limit: self.byte_limit,
            },
        )?;
        let executable = Executable::compile(program.clone(), qos)?;
        while self.entries.len() >= self.capacity || self.bytes > self.byte_limit - bytes {
            let (old, _, _) = self.entries.pop_front().unwrap();
            self.bytes -= old.source_bytes();
        }
        self.entries.push_back((program, qos, executable.clone()));
        self.bytes += bytes;
        Ok(Arc::new(executable))
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn source_bytes(&self) -> usize {
        self.bytes
    }
    pub fn hits(&self) -> usize {
        self.hits
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}
