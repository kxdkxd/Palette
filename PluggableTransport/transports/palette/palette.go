/*
 * Copyright (c) 2014, Yawning Angel <yawning at schwanenlied dot me>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// package palette provides an implementation of the Tor Project's PT Palette
// obfuscation protocol.

package palette

import (
	"encoding/json"
	"fmt"
	"PluggableTransport/common/log"
	"PluggableTransport/common/utils"
	"PluggableTransport/transports/base"
	"PluggableTransport/transports/defconn"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"net"
	"net/http"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"time"

	"git.torproject.org/pluggable-transports/goptlib.git"
)

const (
	transportName = "palette"

	U_uploadArg       = "U_upload"
	U_downloadArg     = "U_download"
	BArg              = "B"
	Alpha_uploadArg   = "Alpha_upload"
	Alpha_downloadArg = "Alpha_download"

	uploadDirection   = 0
	downloadDirection = 1
)

type paletteClientArgs struct {
	*defconn.DefConnClientArgs

	U_upload       int
	U_download     int
	B              int
	Alpha_upload   float32
	Alpha_download float32
}

// Transport is the palette implementation of the base.Transport interface.
type Transport struct {
	defconn.Transport
}

// Name returns the name of the palette transport protocol.
func (t *Transport) Name() string {
	return transportName
}

// ClientFactory returns a new paletteClientFactory instance.
func (t *Transport) ClientFactory(stateDir string) (base.ClientFactory, error) {
	parentFactory, err := t.Transport.ClientFactory(stateDir)
	return &paletteClientFactory{
		parentFactory.(*defconn.DefConnClientFactory),
	}, err
}

// ServerFactory returns a new paletteServerFactory instance.
func (t *Transport) ServerFactory(stateDir string, args *pt.Args) (base.ServerFactory, error) {
	sf, err := t.Transport.ServerFactory(stateDir, args)
	if err != nil {
		return nil, err
	}

	st, err := serverStateFromArgs(stateDir, args)
	if err != nil {
		return nil, err
	}

	paletteSf := paletteServerFactory{
		sf.(*defconn.DefConnServerFactory),
		st.U_upload,
		st.U_download,
		st.B,
		st.Alpha_upload,
		st.Alpha_download,
	}

	return &paletteSf, nil
}

type paletteClientFactory struct {
	*defconn.DefConnClientFactory
}

func (cf *paletteClientFactory) Transport() base.Transport {
	return cf.DefConnClientFactory.Transport()
}

func (cf *paletteClientFactory) ParseArgs(args *pt.Args) (interface{}, error) {
	arguments, err := cf.DefConnClientFactory.ParseArgs(args)

	U_upload, err := utils.GetIntArgFromStr(U_uploadArg, args)
	if err != nil {
		return nil, err
	}
	U_download, err := utils.GetIntArgFromStr(U_downloadArg, args)
	if err != nil {
		return nil, err
	}
	B, err := utils.GetIntArgFromStr(BArg, args)
	if err != nil {
		return nil, err
	}
	Alpha_upload, err := utils.GetFloatArgFromStr(Alpha_uploadArg, args)
	if err != nil {
		return nil, err
	}
	Alpha_download, err := utils.GetFloatArgFromStr(Alpha_downloadArg, args)
	if err != nil {
		return nil, err
	}

	return &paletteClientArgs{
		arguments.(*defconn.DefConnClientArgs),
		U_upload.(int), U_download.(int), B.(int), float32(Alpha_upload.(float64)), float32(Alpha_download.(float64)),
	}, nil
}

func (cf *paletteClientFactory) Dial(network, addr string, dialFn base.DialFunc, args interface{}) (net.Conn, error) {
	defConn, err := cf.DefConnClientFactory.Dial(network, addr, dialFn, args)
	if err != nil {
		return nil, err
	}

	argsT := args.(*paletteClientArgs)
	c := &paletteConn{
		defConn.(*defconn.DefConn),
		0, 100, 300, argsT.U_upload,
		argsT.U_download, argsT.B, argsT.Alpha_upload, argsT.Alpha_download,
		0, 0, 0, 0, false, false,
		false, false, 0, 0, nil, nil, nil, 0, 0,
	}
	return c, nil
}

type paletteServerFactory struct {
	*defconn.DefConnServerFactory
	U_upload       int
	U_download     int
	B              int
	Alpha_upload   float32
	Alpha_download float32
}

func (sf *paletteServerFactory) WrapConn(conn net.Conn) (net.Conn, error) {
	defConn, err := sf.DefConnServerFactory.WrapConn(conn)
	if err != nil {
		return nil, err
	}

	c := &paletteConn{
		defConn.(*defconn.DefConn),
		0, 100, 300, sf.U_upload,
		sf.U_download, sf.B, sf.Alpha_upload, sf.Alpha_download, 0, 0, 0, 0, false,
		false, false, false, 0, 0, nil, nil, nil, 0, 0,
	}
	return c, nil
}

type paletteConn struct {
	*defconn.DefConn
	seqId          int
	uploadParam    int
	downloadParam  int
	U_upload       int
	U_download     int
	B              int
	Alpha_upload   float32
	Alpha_download float32

	nowIdx              int32 // time window index
	totalPacketUpload   uint32
	totalPacketDownload uint32
	ifBranch            int32
	flagEndUpload       bool
	flagEndDownload     bool
	flagFirstUpload     bool
	flagFirstDownload   bool
	thresholdUp         int
	thresholdDown       int
	nowPrunedCenter     [][]int
	uploadKeep          []int
	downloadKeep        []int
	resUp               int32
	resDown             int32
}

func (conn *paletteConn) totalPacketUploadLoad() uint32 {
	return atomic.LoadUint32(&conn.totalPacketUpload)
}
func (conn *paletteConn) totalPacketDownloadLoad() uint32 {
	return atomic.LoadUint32(&conn.totalPacketDownload)
}
func (conn *paletteConn) totalPacketUploadIncrement() {
	atomic.AddUint32(&conn.totalPacketUpload, 1)
}
func (conn *paletteConn) totalPacketDownloadIncrement() {
	atomic.AddUint32(&conn.totalPacketDownload, 1)
}
func (conn *paletteConn) totalPacketCountReset() {
	atomic.StoreUint32(&conn.totalPacketUpload, 0)
	atomic.StoreUint32(&conn.totalPacketDownload, 0)
}

func (conn *paletteConn) ReadFrom(r io.Reader) (written int64, err error) {
	if conn.IsServer {
		return conn.ReadFromAtServer(r)
	} else {
		return conn.ReadFromAtClient(r)
	}
}

func (conn *paletteConn) Read(b []byte) (n int, err error) {
	return conn.DefConn.MyRead(b, conn.readPackets)
}

func (conn *paletteConn) ReadFromAtClient(r io.Reader) (written int64, err error) {
	log.Infof("[State] PALETTE Enter copyloop state: %v at %v", defconn.StateMap[conn.ConnState.LoadCurState()], time.Now().Format("15:04:05.000"))
	defer close(conn.CloseChan)

	var receiveBuf utils.SafeBuffer //read payload from upstream and buffer here

	go conn.Send()

	// this go routine regularly check the real throughput
	// if it is small, change to stop state
	go func() {
		ticker := time.NewTicker(defconn.TWindow) // 4s
		defer ticker.Stop()
		for {
			select {
			case _, ok := <-conn.CloseChan:
				if !ok {
					log.Infof("[Routine] Ticker routine exits by closeChan.")
					return
				}
			case <-ticker.C:
				log.Debugf("[State] Real Sent: %v, Real Receive: %v, curState: %s at %v.", conn.NRealSegSentLoad(), conn.NRealSegRcvLoad(), defconn.StateMap[conn.ConnState.LoadCurState()], time.Now().Format("15:04:05.000000"))
				if !conn.IsServer && conn.ConnState.LoadCurState() != defconn.StateStop && (conn.NRealSegSentLoad() < 2 || conn.NRealSegRcvLoad() < 2) {
					log.Infof("[State] %s -> %s.", defconn.StateMap[conn.ConnState.LoadCurState()], defconn.StateMap[defconn.StateStop])
					conn.ConnState.SetState(defconn.StateStop)
					conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeSignalStop, Data: []byte{}, PadLen: defconn.MaxPacketPaddingLength}
					conn.totalPacketCountReset()
				}
				conn.NRealSegReset()
			}
		}
	}()

	go func() {
		for {
			select {
			case _, ok := <-conn.CloseChan:
				if !ok {
					log.Infof("[Routine] Send routine exits by closedChan.")
					return
				}
			default:
				buf := make([]byte, 65535)
				rdLen, err := r.Read(buf[:])
				if err != nil {
					log.Errorf("Exit by read err:%v", err)
					conn.ErrChan <- err
					return
				}
				if rdLen > 0 {
					_, err := receiveBuf.Write(buf[:rdLen])
					if err != nil {
						conn.ErrChan <- err
						return
					}
					// signal server to start if there is more than one cell coming
					// else switch to padding state
					// stop -> ready -> start
					if (conn.ConnState.LoadCurState() == defconn.StateStop && rdLen > defconn.MaxPacketPayloadLength) ||
						(conn.ConnState.LoadCurState() == defconn.StateReady) {
						// stateStop with >2 cells -> stateStart
						// or stateReady with >0 cell -> stateStart
						if atomic.LoadInt32(&conn.ifBranch) == 1 {
							log.Infof("[State] Got %v bytes upstream, %s -> %s.", rdLen, defconn.StateMap[conn.ConnState.LoadCurState()], defconn.StateMap[defconn.StateStart])
							conn.ConnState.SetState(defconn.StateStart)
							atomic.StoreInt32(&conn.nowIdx, 0)
							conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeSignalStart, Data: []byte{byte(conn.seqId)}, PadLen: defconn.MaxPacketPaddingLength - 1}
						}
					} else if conn.ConnState.LoadCurState() == defconn.StateStop {
						log.Infof("[State] Got %v bytes upstream, %s -> %s.", rdLen, defconn.StateMap[defconn.StateStop], defconn.StateMap[defconn.StateReady])
						conn.ConnState.SetState(defconn.StateReady)
					}
				} else {
					log.Errorf("BUG? read 0 bytes, err: %v", err)
					conn.ErrChan <- io.EOF
					return
				}
			}
		}
	}()

	http.HandleFunc("/enter", func(w http.ResponseWriter, r *http.Request) { // start defense
		log.Noticef("[Palette] Enter Defense Mode")
		params := r.URL.Query()
		seqIdStr := params.Get("seqId")
		if seqIdStr == "" {
			_, err := w.Write([]byte("Missing seqId parameter"))
			if err != nil {
				return
			}
			return
		}
		seqId, err := strconv.Atoi(seqIdStr)
		if err != nil {
			_, err := w.Write([]byte("Invalid seqId parameter"))
			if err != nil {
				return
			}
			return
		}
		conn.seqId = seqId
		conn.flagEndUpload = false
		conn.flagFirstUpload = false
		conn.totalPacketCountReset()
		conn.thresholdUp = rand.Intn(conn.U_upload)
		conn.nowPrunedCenter, conn.uploadKeep, _ = conn.getStrategy(uploadDirection, conn.seqId)
		atomic.StoreInt32(&conn.resUp, 0)
		atomic.StoreInt32(&conn.nowIdx, 0)
		atomic.StoreInt32(&conn.ifBranch, 1)
		_, err = w.Write([]byte("Enter operation completed successfully"))
		if err != nil {
			return
		}
	})

	http.HandleFunc("/exit", func(w http.ResponseWriter, r *http.Request) { // stop defense
		log.Noticef("[Palette] Exit Defense Mode")
		atomic.StoreInt32(&conn.ifBranch, 0)
		conn.ConnState.SetState(defconn.StateStop)
		conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeSignalStop, Data: []byte{}, PadLen: defconn.MaxPacketPaddingLength}
		conn.totalPacketCountReset()
		_, err := w.Write([]byte("Exit operation completed successfully"))
		if err != nil {
			return
		}
	})

	go func() {
		err := http.ListenAndServe(":7999", nil)
		if err != nil {
			return
		}
	}()

	///////////////////////////////////////////////////////////////////
	log.Noticef("[Palette] Init")
	atomic.StoreInt32(&conn.nowIdx, 0)
	conn.totalPacketCountReset()
	conn.thresholdUp = rand.Intn(conn.U_upload)
	conn.nowPrunedCenter, conn.uploadKeep, _ = conn.getStrategy(uploadDirection, conn.seqId)
	log.Debugf("[Palette] nowPrunedCenter=%v,uploadKeep=%v", conn.nowPrunedCenter, conn.uploadKeep)
	atomic.StoreInt32(&conn.resUp, 0)
	for {
		select {
		case conErr := <-conn.ErrChan:
			log.Infof("upload copy loop terminated at %v. Reason: %v", time.Now().Format("15:04:05.000000"), conErr)
			return written, conErr
		default:
			nowIdx := atomic.LoadInt32(&conn.nowIdx)
			if (conn.ConnState.LoadCurState() == defconn.StateStart) && nowIdx < 1000 && atomic.LoadInt32(&conn.ifBranch) == 1 { //defense on, nowIdx 1000 means 80seconds
				startTime := time.Now()
				log.Debugf("[Palette] nowIdx=%v", nowIdx)
				rand.Seed(time.Now().UnixNano())
				lastUpload := 0
				for i := nowIdx; i < nowIdx+int32(conn.thresholdUp) && i < int32(len(conn.nowPrunedCenter[0])); i++ {
					lastUpload += conn.nowPrunedCenter[0][i]
				}
				uploadCenter := conn.nowPrunedCenter[0][nowIdx]
				uploadTarget := uploadCenter

				uploadReal := uploadTarget

				if contains(conn.uploadKeep, int(nowIdx)) {
					nxtLenUp := 1
					for !contains(conn.uploadKeep, nxtLenUp+int(nowIdx)) && nxtLenUp+int(nowIdx) < 1000 {
						nxtLenUp++
					}
					atomic.StoreInt32(&conn.resUp, int32(math.Ceil(float64(uploadTarget)/float64(nxtLenUp))))
				}
				uploadReal = int(atomic.LoadInt32(&conn.resUp))

				nowTimePacketNum := int(math.Ceil(float64(receiveBuf.GetLen()) / float64(defconn.MaxPacketPayloadLength)))
				if nowTimePacketNum != 0 {
					conn.flagEndUpload = false
				}
				sendUploadPacketCount := uploadReal
				if nowTimePacketNum == 0 {
					if conn.flagEndUpload {
						sendUploadPacketCount = 0
					} else if nowIdx != 0 && nowIdx%int32(conn.B) == 0 {
						conn.flagEndUpload = true
					}
				}
				if conn.totalPacketUploadLoad() >= 1 && conn.flagFirstUpload {
					if nowTimePacketNum != 0 && nowTimePacketNum >= lastUpload {
						sendUploadPacketCount = int(math.Max(math.Min(float64(conn.uploadParam), float64(nowTimePacketNum)), float64(uploadTarget)))
						log.Debugf("[Palette] sendUploadPacketCount=%v， conn.totalPacketUploadLoad()=%v, flagFirstUpload=%v", sendUploadPacketCount, conn.totalPacketUploadLoad(), conn.flagFirstUpload)
					}
				} else { // first round
					if nowTimePacketNum != 0 {
						sendUploadPacketCount = int(math.Max(float64(nowTimePacketNum), float64(uploadReal)))
						log.Debugf("[Palette] sendUploadPacketCount=%v, conn.totalPacketUploadLoad()=%v, flagFirstUpload=%v, nowTimePacketNum=%v, uploadReal=%v", sendUploadPacketCount, conn.totalPacketUploadLoad(), conn.flagFirstUpload, nowTimePacketNum, uploadReal)
					} else {
						sendUploadPacketCount = 0
					}
				}

				log.Debugf("[Palette] sendUploadPacketCount=%v", sendUploadPacketCount)

				for sendUploadPacketCount > 0 {
					var payload [defconn.MaxPacketPayloadLength]byte
					rdLen, _ := receiveBuf.Read(payload[:]) // block if no data
					written += int64(rdLen)
					var pktType uint8
					if rdLen > 0 {
						pktType = defconn.PacketTypePayload
						conn.NRealSegSentIncrement()
						conn.totalPacketUploadIncrement()
					} else {
						// no data, send out a dummy packet
						pktType = defconn.PacketTypeDummy
					}
					conn.SendChan <- defconn.PacketInfo{PktType: pktType, Data: payload[:rdLen], PadLen: uint16(defconn.MaxPacketPaddingLength - rdLen)}
					sendUploadPacketCount -= 1
				}
				if conn.totalPacketUploadLoad() >= 1 {
					conn.flagFirstUpload = true
				}
				atomic.StoreInt32(&conn.nowIdx, nowIdx+1)
				elapsedMicroseconds := time.Since(startTime).Microseconds()
				sleepTime := 80*int64(time.Millisecond) - elapsedMicroseconds*int64(time.Microsecond)

				if sleepTime > 0 {
					time.Sleep(time.Duration(sleepTime))
				} else {
					log.Warnf("[Palette] Execution time exceeded 80 milliseconds, skipping sleep.")
				}
			} else {
				//defense off (in stop or ready)
				//log.Noticef("[Palette] defense off (in stop or ready)")
				writtenTmp, werr := conn.sendRealBurst(&receiveBuf, conn.SendChan)
				written += writtenTmp
				if werr != nil {
					return written, werr
				}
				time.Sleep(50 * time.Millisecond) // avoid infinite loop
			}
		}
	}
	///////////////////////////////////////////////////////////////////
}

func (conn *paletteConn) ReadFromAtServer(r io.Reader) (written int64, err error) {
	log.Infof("[State] PALETTE Enter copyloop state: %v at %v", defconn.StateMap[conn.ConnState.LoadCurState()], time.Now().Format("15:04:05.000"))
	defer close(conn.CloseChan)

	var receiveBuf utils.SafeBuffer // read payload from upstream and buffer here

	// create a go routine to send out packets to the wire
	go conn.Send()

	go func() {
		for {
			select {
			case _, ok := <-conn.CloseChan:
				if !ok {
					log.Infof("[Routine] Send routine exits by closedChan.")
					return
				}
			default:
				buf := make([]byte, 65535)
				rdLen, err := r.Read(buf[:])
				if err != nil {
					log.Errorf("Exit by read err:%v", err)
					conn.ErrChan <- err
					return
				}
				if rdLen > 0 {
					_, err := receiveBuf.Write(buf[:rdLen])
					if err != nil {
						conn.ErrChan <- err
						return
					}
				} else {
					log.Errorf("BUG? uread 0 bytes, err: %v", err)
					conn.ErrChan <- io.EOF
					return
				}
			}
		}
	}()

	///////////////////////////////////////////////////////////////////
	log.Noticef("[Palette] Init")
	atomic.StoreInt32(&conn.nowIdx, 0)
	conn.flagEndDownload = false
	conn.flagFirstDownload = false
	conn.nowPrunedCenter, _, conn.downloadKeep = conn.getStrategy(downloadDirection, conn.seqId)
	log.Debugf("[Palette] nowPrunedCenter=%v,downloadKeep=%v", conn.nowPrunedCenter, conn.downloadKeep)
	conn.totalPacketCountReset()
	atomic.StoreInt32(&conn.resDown, 0)
	conn.thresholdDown = rand.Intn(conn.U_download) + 1

	for {
		select {
		case conErr := <-conn.ErrChan:
			log.Infof("downstream copy loop terminated at %v. Reason: %v", time.Now().Format("15:04:05.000000"), conErr)
			return written, conErr
		default:
			nowIdx := atomic.LoadInt32(&conn.nowIdx)
			if (conn.ConnState.LoadCurState() == defconn.StateStart) && nowIdx < 1000 { //defense on, nowIdx 1000 means 80seconds
				startTime := time.Now()
				log.Debugf("[Palette] nowIdx=%v", nowIdx)
				rand.Seed(time.Now().UnixNano())

				lastDownload := 0
				for i := nowIdx; i < nowIdx+int32(conn.thresholdDown) && i < int32(len(conn.nowPrunedCenter[1])); i++ {
					lastDownload += conn.nowPrunedCenter[1][i]
				}
				downloadCenter := conn.nowPrunedCenter[1][nowIdx]
				downloadTarget := downloadCenter

				downloadReal := downloadTarget

				if contains(conn.downloadKeep, int(nowIdx)) {
					nxtLenDown := 1
					for !contains(conn.downloadKeep, nxtLenDown+int(nowIdx)) && nxtLenDown+int(nowIdx) < 1000 {
						nxtLenDown++
					}
					atomic.StoreInt32(&conn.resDown, int32(math.Ceil(float64(downloadTarget)/float64(nxtLenDown))))
				}
				downloadReal = int(atomic.LoadInt32(&conn.resDown))

				nowTimePacketNum := int(math.Ceil(float64(receiveBuf.GetLen()) / float64(defconn.MaxPacketPayloadLength)))
				if nowTimePacketNum != 0 {
					conn.flagEndDownload = false
				}
				sendDownloadPacketCount := downloadReal
				if nowTimePacketNum == 0 {
					if conn.flagEndDownload {
						sendDownloadPacketCount = 0
					} else if nowIdx != 0 && nowIdx%int32(conn.B) == 0 {
						conn.flagEndDownload = true
					}
				}
				if conn.totalPacketDownloadLoad() >= 1 && conn.flagFirstDownload {
					if nowTimePacketNum != 0 && nowTimePacketNum >= lastDownload {
						sendDownloadPacketCount = int(math.Max(math.Min(float64(conn.downloadParam), float64(nowTimePacketNum)), float64(downloadTarget)))
						log.Debugf("[Palette] sendDownloadPacketCount=%v， conn.totalPacketDownloadLoad()=%v, flagFirstDownload=%v", sendDownloadPacketCount, conn.totalPacketDownloadLoad(), conn.flagFirstDownload)
					}
				} else { // First Round
					if nowTimePacketNum != 0 {
						sendDownloadPacketCount = int(math.Max(float64(nowTimePacketNum), float64(downloadReal)))
						log.Debugf("[Palette] sendDownloadPacketCount=%v, conn.totalPacketDownloadLoad()=%v, flagFirstDownload=%v, nowTimePacketNum=%v, downloadReal=%v", sendDownloadPacketCount, conn.totalPacketDownloadLoad(), conn.flagFirstDownload, nowTimePacketNum, downloadReal)
					} else {
						sendDownloadPacketCount = 0
					}
				}

				log.Debugf("[Palette] sendDownloadPacketCount=%v", sendDownloadPacketCount)

				for sendDownloadPacketCount > 0 {
					var payload [defconn.MaxPacketPayloadLength]byte

					rdLen, _ := receiveBuf.Read(payload[:])
					written += int64(rdLen)
					var pktType uint8
					if rdLen > 0 {
						pktType = defconn.PacketTypePayload
						conn.totalPacketDownloadIncrement()
					} else {
						// no data, send out a dummy packet
						pktType = defconn.PacketTypeDummy
					}
					conn.SendChan <- defconn.PacketInfo{PktType: pktType, Data: payload[:rdLen], PadLen: uint16(defconn.MaxPacketPaddingLength - rdLen)}

					sendDownloadPacketCount -= 1
				}

				if conn.totalPacketDownloadLoad() >= 1 {
					conn.flagFirstDownload = true
				}

				atomic.StoreInt32(&conn.nowIdx, nowIdx+1)
				elapsedMicroseconds := time.Since(startTime).Microseconds()
				sleepTime := 80*int64(time.Millisecond) - elapsedMicroseconds*int64(time.Microsecond)

				if sleepTime > 0 {
					time.Sleep(time.Duration(sleepTime))
				} else {
					log.Warnf("[Palette] Execution time exceeded 80 milliseconds, skipping sleep.")
				}
			} else {
				//defense off (in stop or ready)
				//log.Noticef("[Palette] defense off (in stop or ready)")
				writtenTmp, werr := conn.sendRealBurst(&receiveBuf, conn.SendChan)
				written += writtenTmp
				if werr != nil {
					log.Debugf("[Palette] written=%v, werr=%v", written, werr)
					return written, werr
				}
				time.Sleep(50 * time.Millisecond) //avoid infinite loop
			}
		}
	}
}

///////////////////////////////////////////////////////////////////

func (conn *paletteConn) sendRealBurst(receiveBuf *utils.SafeBuffer, sendChan chan defconn.PacketInfo) (written int64, err error) {
	if size := receiveBuf.GetLen(); size > 0 {
		log.Debugf("[OFF] Send %v bytes", size)
	}
	for receiveBuf.GetLen() > 0 {
		var payload [defconn.MaxPacketPayloadLength]byte
		rdLen, rdErr := receiveBuf.Read(payload[:])
		written += int64(rdLen)
		if rdErr != nil {
			log.Infof("Exit by read buffer err:%v", rdErr)
			return written, rdErr
		}
		sendChan <- defconn.PacketInfo{PktType: defconn.PacketTypePayload, Data: payload[:rdLen], PadLen: uint16(defconn.MaxPacketPaddingLength - rdLen)}
		if !conn.IsServer {
			conn.NRealSegSentIncrement()
		}
	}
	return
}

func readJSONFile(filename string, v interface{}) error {
	absPath, err := filepath.Abs(filename)
	if err != nil {
		return fmt.Errorf("failed to get absolute path: %w", err)
	}
	log.Debugf("[Palette] Absolute path: %s", absPath)

	fileContent, err := ioutil.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("file reading error: %w", err)
	}

	contentStr := string(fileContent)
	if len(contentStr) > 100 {
		log.Debugf("Content of %s: %s...", filename, contentStr[:100])
	} else {
		log.Debugf("Content of %s: %s", filename, contentStr)
	}

	if err := json.Unmarshal(fileContent, v); err != nil {
		return fmt.Errorf("error while decoding JSON: %w", err)
	}

	return nil
}

func (conn *paletteConn) getStrategy(direction int, seqId int) ([][]int, []int, []int) {
	var prunedCenterSet [][][]int
	var traceProbUpload [][]float64
	var traceProbDownload [][]float64
	log.Infof("[Palette] getStrategy,seqId=%v", seqId)
	if err := readJSONFile("prunedCenterSet.json", &prunedCenterSet); err != nil {
		log.Errorf("Error reading prunedCenterSet: %v", err)
	}

	if err := readJSONFile("traceProbUpload.json", &traceProbUpload); err != nil {
		log.Errorf("Error reading traceProbUpload: %v", err)
	}

	if err := readJSONFile("traceProbDownload.json", &traceProbDownload); err != nil {
		log.Errorf("Error reading traceProbDownload: %v", err)
	}
	// Now GetKRandom Here
	tamLen := 1000 // 1000 slots in TAM
	rand.Seed(time.Now().UnixNano())
	randomList := rand.Perm(tamLen)

	uploadWhereToKeep := make([]int, 0)
	downloadWhereToKeep := make([]int, 0)
	sumThreshold := 0.0
	traceCumResUpload := traceProbUpload[seqId]
	traceCumResDownload := traceProbDownload[seqId]

	if direction == uploadDirection { // upload
		threshold := float64(conn.Alpha_upload)
		log.Debugf("[Palette] threshold=%v,seqId=%v,direction=%v", threshold, seqId, direction)
		for i := 0; i < tamLen; i++ {
			if sumThreshold >= threshold {
				break
			}
			sumThreshold += traceCumResUpload[randomList[i]]
			uploadWhereToKeep = append(uploadWhereToKeep, randomList[i])
		}
	} else if direction == downloadDirection { // download
		threshold := float64(conn.Alpha_download)
		log.Debugf("[Palette] threshold=%v,seqId=%v,direction=%v", threshold, seqId, direction)
		sumThreshold = 0
		for i := 0; i < tamLen; i++ {
			if sumThreshold >= threshold {
				break
			}
			sumThreshold += traceCumResDownload[randomList[i]]
			downloadWhereToKeep = append(downloadWhereToKeep, randomList[i])
		}
	}
	return prunedCenterSet[seqId], uploadWhereToKeep, downloadWhereToKeep
}

func contains(slice []int, value int) bool {
	for _, v := range slice {
		if v == value {
			return true
		}
	}
	return false
}

var _ base.ClientFactory = (*paletteClientFactory)(nil)
var _ base.ServerFactory = (*paletteServerFactory)(nil)
var _ base.Transport = (*Transport)(nil)
var _ net.Conn = (*paletteConn)(nil)
