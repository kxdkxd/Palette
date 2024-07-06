package utils

import (
	"bytes"
	"sync"
)

type SafeBuffer struct {
	buffer bytes.Buffer
	mutex  sync.RWMutex
}

func (s *SafeBuffer) Write(p []byte) (n int, err error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return s.buffer.Write(p)
}

func (s *SafeBuffer) Read(p []byte) (n int, err error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	return s.buffer.Read(p)
}

func (s *SafeBuffer) GetLen() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	return s.buffer.Len()
}
