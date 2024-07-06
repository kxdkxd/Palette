package utils

import (
	"math"
	"testing"
)

func TestReadFloatFromFile(t *testing.T) {
	arr :=ReadFloatFromFile("../../transports/wfgan/grpc/time_feature_0-100x0-1000_o2o.ipt")
	if math.Abs(arr[0] - 0.002031) >= 1e-6 {
		t.Fatal("error")
	}
	if len(arr) != 10000000+1 {
		t.Fatal("Error")
	}
}


