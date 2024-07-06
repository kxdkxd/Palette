package utils

import (
	"bufio"
	"fmt"
	pt "git.torproject.org/pluggable-transports/goptlib.git"
	"github.com/mikioh/tcp"
	"github.com/mikioh/tcpinfo"
	"PluggableTransport/common/log"
	"net"

	//"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func SleepRho(lastSend time.Time, rho time.Duration)  {
	deltaT := time.Now().Sub(lastSend)
	if remainingDelay := rho - deltaT; remainingDelay > 0 {
		// We got data faster than the pacing rate, sleep
		// for the remaining time.
		time.Sleep(remainingDelay)
	}
}


func ParseArgByKey(args *pt.Args, key string, kind string) (interface{}, error) {
	kind = strings.ToLower(kind)
	Str, Ok := args.Get(key)
	if !Ok {
		return nil, fmt.Errorf("missing argument '%s'", key)
	}
	if kind == "int" {
		arg, err := strconv.Atoi(Str)
		if err != nil {
			return nil, fmt.Errorf("malformed '%s': '%s'", key, Str)
		}
		return arg, nil
	} else if kind == "float32" || kind == "float64" {
		precision, err := strconv.Atoi(kind[5:5+2])
		if err != nil {
			return nil, err
		}
		arg, err := strconv.ParseFloat(Str, precision)
		if err != nil {
			return nil, fmt.Errorf("malformed '%s': '%s'", key, Str)
		}
		return arg, nil
	}
	return nil, fmt.Errorf("wrong kind: '%s'", kind)
}


func ReadFloatFromFile(fdir string) []float64{
	file, err := os.Open(fdir)
	if err != nil {
		log.Errorf("[ERR] Fail to open file: %v", err)
	}
	defer file.Close()
	var arr []float64
	scanner := bufio.NewScanner(file)
	startTime := time.Now()
	for scanner.Scan() {
		lineStr := scanner.Text()
		num, sErr := strconv.ParseFloat(lineStr, 64)
		if sErr == nil {
			arr = append(arr,num)
		}
	}
	log.Debugf("Successfully loading %v ipt info in %v", len(arr), time.Since(startTime))
	return arr
}


func SampleIPT(arr []float64) float64{
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	std := arr[0]
	l := len(arr) -1
	ind := r.Intn(l) + 1
	res := arr[ind] + r.NormFloat64()*std
	res = math.Pow(10, res)
	res *= 1000  // seconds to milliseconds
	if res < 0 {
		//negative ipt
		res = 0
	}
	return res
}

func EstimateTCPCapacity(conn net.Conn) (capacity int, err error){
	tc, err := tcp.NewConn(conn)
	if err != nil {
		return 0, err
	}

	var o tcpinfo.Info
	var b [256]byte
	i, err := tc.Option(o.Level(), o.Name(), b[:])
	if j, ok := i.(*tcpinfo.Info); ok{
		snd_cwnd := int(j.CongestionControl.SenderWindowSegs)
		//unacked := int(j.Sys.UnackedSegs)
		unacked := 0 //macos no such info

		capacity = (snd_cwnd - unacked) * int(j.SenderMSS)
		return capacity, nil
	}


	return 0, err
}


func GetIntArgFromStr(argName string, args *pt.Args) (interface {}, error) {
	argStr, argOK := args.Get(argName)
	if !argOK {
		return nil, fmt.Errorf("missing argument '%s'", argName)
	}
	argValue, err := strconv.Atoi(argStr)
	if err != nil {
		return nil, fmt.Errorf("malformed '%s' '%s'", argName, argStr)
	}
	return argValue, nil
}

func GetFloatArgFromStr(argName string, args *pt.Args) (interface {}, error) {
	argStr, argOK := args.Get(argName)
	if !argOK {
		return nil, fmt.Errorf("missing argument '%s'", argName)
	}
	argValue, err := strconv.ParseFloat(argStr, 64)
	if err != nil {
		return nil, fmt.Errorf("malformed '%s' '%s'", argName, argStr)
	}
	return argValue, nil
}

func GetFormattedCurrentTime(t time.Time) string {
	return t.Format("15:04:05.000000")
}
