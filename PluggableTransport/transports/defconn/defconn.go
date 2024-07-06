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

// package defconn provides an implementation of the Tor Project's defconn
// obfuscation protocol.
package defconn // import "PluggableTransport/transports/defconn"

import (
	"bytes"
	"fmt"
	"PluggableTransport/common/log"
	"PluggableTransport/common/utils"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"syscall"
	"time"

	"git.torproject.org/pluggable-transports/goptlib.git"
	"PluggableTransport/common/drbg"
	"PluggableTransport/common/ntor"
	"PluggableTransport/common/probdist"
	"PluggableTransport/common/replayfilter"
	"PluggableTransport/transports/base"
	"PluggableTransport/transports/defconn/framing"
)

const (
	transportName = "defconn"

	nodeIDArg     = "node-id"
	publicKeyArg  = "public-key"
	privateKeyArg = "private-key"
	seedArg       = "drbg-seed"
	certArg       = "cert"

	seedLength             = drbg.SeedLength
	headerLength           = framing.FrameOverhead + PacketOverhead
	clientHandshakeTimeout = time.Duration(60) * time.Second
	serverHandshakeTimeout = time.Duration(30) * time.Second
	replayTTL              = time.Duration(3) * time.Hour

	maxCloseDelay = 60
	TWindow       = 4000 * time.Millisecond

	LogEnabled = true
)

type ClientArgs interface {
	GetNodeID() *ntor.NodeID
	GetPublicKey() *ntor.PublicKey
	GetSessionKey() *ntor.Keypair
}

type DefConnClientArgs struct {
	nodeID     *ntor.NodeID
	publicKey  *ntor.PublicKey
	sessionKey *ntor.Keypair
}

func (d *DefConnClientArgs) GetNodeID() *ntor.NodeID {
	return d.nodeID
}

func (d *DefConnClientArgs) GetPublicKey() *ntor.PublicKey {
	return d.publicKey
}

func (d *DefConnClientArgs) GetSessionKey() *ntor.Keypair {
	return d.sessionKey
}

// Transport is the defconn implementation of the base.DefTransport interface.
type Transport struct{}

// Name returns the name of the defconn transport protocol.
func (t *Transport) Name() string {
	return transportName
}

// ClientFactory returns a new DefConnClientFactory instance.
func (t *Transport) ClientFactory(stateDir string) (base.ClientFactory, error) {
	cf := &DefConnClientFactory{transport: t}
	return cf, nil
}

// ServerFactory returns a new DefConnServerFactory instance.
func (t *Transport) ServerFactory(stateDir string, args *pt.Args) (base.ServerFactory, error) {
	st, err := ServerStateFromArgs(stateDir, args)
	if err != nil {
		return nil, err
	}

	// Store the arguments that should appear in our descriptor for the clients.
	ptArgs := pt.Args{}
	ptArgs.Add(certArg, st.cert.String())

	// Initialize the replay filter.
	filter, err := replayfilter.New(replayTTL)
	if err != nil {
		return nil, err
	}

	// Initialize the close thresholds for failed connections.
	drbg, err := drbg.NewHashDrbg(st.drbgSeed)
	if err != nil {
		return nil, err
	}
	rng := rand.New(drbg)

	sf := &DefConnServerFactory{t, &ptArgs, st.nodeID, st.identityKey, st.drbgSeed, filter, rng.Intn(maxCloseDelay)}
	return sf, nil
}

type DefConnClientFactory struct {
	transport base.Transport
}

func (cf *DefConnClientFactory) Transport() base.Transport {
	return cf.transport
}

func (cf *DefConnClientFactory) ParseArgs(args *pt.Args) (interface{}, error) {
	var nodeID *ntor.NodeID
	var publicKey *ntor.PublicKey

	// The "new" (version >= 0.0.3) bridge lines use a unified "cert" argument
	// for the Node ID and Public Key.
	certStr, ok := args.Get(certArg)
	if ok {
		cert, err := serverCertFromString(certStr)
		if err != nil {
			return nil, err
		}
		nodeID, publicKey = cert.unpack()
	} else {
		// The "old" style (version <= 0.0.2) bridge lines use separate Node ID
		// and Public Key arguments in Base16 encoding and are a UX disaster.
		nodeIDStr, ok := args.Get(nodeIDArg)
		if !ok {
			return nil, fmt.Errorf("missing argument '%s'", nodeIDArg)
		}
		var err error
		if nodeID, err = ntor.NodeIDFromHex(nodeIDStr); err != nil {
			return nil, err
		}

		publicKeyStr, ok := args.Get(publicKeyArg)
		if !ok {
			return nil, fmt.Errorf("missing argument '%s'", publicKeyArg)
		}
		if publicKey, err = ntor.PublicKeyFromHex(publicKeyStr); err != nil {
			return nil, err
		}
	}

	// Generate the session key pair before connectiong to hide the Elligator2
	// rejection sampling from network observers.
	sessionKey, err := ntor.NewKeypair(true)
	if err != nil {
		return nil, err
	}

	return &DefConnClientArgs{nodeID, publicKey, sessionKey}, nil
}

func (cf *DefConnClientFactory) Dial(network, addr string, dialFn base.DialFunc, args interface{}) (net.Conn, error) {
	// Validate args before bothering to open connection.
	ca, ok := args.(ClientArgs)
	if !ok {
		return nil, fmt.Errorf("invalid argument type for args")
	}
	conn, err := dialFn(network, addr)
	if err != nil {
		return nil, err
	}
	dialConn := conn
	if conn, err = newdefconnClientConn(conn, ca); err != nil {
		dialConn.Close()
		return nil, err
	}
	return conn, nil
}

type DefConnServerFactory struct {
	transport base.Transport
	args      *pt.Args

	nodeID      *ntor.NodeID
	identityKey *ntor.Keypair
	lenSeed     *drbg.Seed

	replayFilter *replayfilter.ReplayFilter
	closeDelay   int
}

func (sf *DefConnServerFactory) Transport() base.Transport {
	return sf.transport
}

func (sf *DefConnServerFactory) Args() *pt.Args {
	return sf.args
}

func (sf *DefConnServerFactory) WrapConn(conn net.Conn) (net.Conn, error) {
	// Not much point in having a separate newdefconnServerConn routine when
	// wrapping requires using values from the factory instance.

	// Generate the session keypair *before* consuming Data from the peer, to
	// attempt to mask the rejection sampling due to use of Elligator2.  This
	// might be futile, but the timing differential isn't very large on modern
	// hardware, and there are far easier statistical attacks that can be
	// mounted as a distinguisher.
	sessionKey, err := ntor.NewKeypair(true)
	if err != nil {
		return nil, err
	}

	lenDist := probdist.New(sf.lenSeed, 0, framing.MaximumSegmentLength, false)

	closeChan := make(chan int)
	errChan := make(chan error, 10)
	sendChan := make(chan PacketInfo, 10000)

	// The server's initial state is intentionally set to stateStart at the very beginning to obfuscate the RTT between client and server
	c := &DefConn{conn, true, lenDist, 0, 0, NewState(), closeChan, errChan, sendChan, bytes.NewBuffer(nil), bytes.NewBuffer(nil), make([]byte, ConsumeReadSize), nil, nil}
	startTime := time.Now()

	if err = c.serverHandshake(sf, sessionKey); err != nil {
		log.Errorf("Handshake err %v", err)
		c.closeAfterDelay(sf, startTime)
		return nil, err
	}

	return c, nil
}

type DefConn struct {
	net.Conn

	IsServer bool
	LenDist  *probdist.WeightedDist

	nRealSegSent uint32 // real packet counter over TWindow second
	nRealSegRcv  uint32
	ConnState    *State

	CloseChan chan int        //used to signal go routines for stop
	ErrChan   chan error      //used to send err between go routines
	SendChan  chan PacketInfo //used to send packets to the send routine

	ReceiveBuffer        *bytes.Buffer
	ReceiveDecodedBuffer *bytes.Buffer
	ReadBuffer           []byte

	Encoder *framing.Encoder
	Decoder *framing.Decoder
}

func newdefconnClientConn(conn net.Conn, args ClientArgs) (c *DefConn, err error) {
	// Generate the initial protocol polymorphism distribution(s).
	var seed *drbg.Seed
	if seed, err = drbg.NewSeed(); err != nil {
		return
	}
	lenDist := probdist.New(seed, 0, framing.MaximumSegmentLength, false)

	closeChan := make(chan int)
	errChan := make(chan error, 10)
	sendChan := make(chan PacketInfo, 10000)

	// Allocate the client structure.
	c = &DefConn{conn, false, lenDist, 0, 0, NewState(), closeChan, errChan, sendChan, bytes.NewBuffer(nil), bytes.NewBuffer(nil), make([]byte, ConsumeReadSize), nil, nil}

	// Start the handshake timeout.
	deadline := time.Now().Add(clientHandshakeTimeout)
	if err = conn.SetDeadline(deadline); err != nil {
		return nil, err
	}

	if err = c.clientHandshake(args.GetNodeID(), args.GetPublicKey(), args.GetSessionKey()); err != nil {
		return nil, err
	}

	// Stop the handshake timeout.
	if err = conn.SetDeadline(time.Time{}); err != nil {
		return nil, err
	}

	return
}

func (conn *DefConn) clientHandshake(nodeID *ntor.NodeID, peerIdentityKey *ntor.PublicKey, sessionKey *ntor.Keypair) error {
	if conn.IsServer {
		return fmt.Errorf("clientHandshake called on server connection")
	}

	// Generate and send the client handshake.
	hs := newClientHandshake(nodeID, peerIdentityKey, sessionKey)
	blob, err := hs.generateHandshake()
	if err != nil {
		return err
	}
	if _, err = conn.Conn.Write(blob); err != nil {
		return err
	}

	// Consume the server handshake.
	var hsBuf [maxHandshakeLength]byte
	for {
		n, err := conn.Conn.Read(hsBuf[:])
		if err != nil {
			// The Read() could have returned Data and an error, but there is
			// no point in continuing on an EOF or whatever.
			return err
		}
		conn.ReceiveBuffer.Write(hsBuf[:n])

		n, seed, err := hs.parseServerHandshake(conn.ReceiveBuffer.Bytes())
		if err == ErrMarkNotFoundYet {
			continue
		} else if err != nil {
			return err
		}
		_ = conn.ReceiveBuffer.Next(n)

		// Use the derived key material to intialize the link crypto.
		okm := ntor.Kdf(seed, framing.KeyLength*2)
		conn.Encoder = framing.NewEncoder(okm[:framing.KeyLength])
		conn.Decoder = framing.NewDecoder(okm[framing.KeyLength:])

		return nil
	}
}

func (conn *DefConn) serverHandshake(sf *DefConnServerFactory, sessionKey *ntor.Keypair) error {
	if !conn.IsServer {
		return fmt.Errorf("serverHandshake called on client connection")
	}

	// Generate the server handshake, and arm the base timeout.
	hs := newServerHandshake(sf.nodeID, sf.identityKey, sessionKey)
	if err := conn.Conn.SetDeadline(time.Now().Add(serverHandshakeTimeout)); err != nil {
		return err
	}

	// Consume the client handshake.
	var hsBuf [maxHandshakeLength]byte
	for {
		n, err := conn.Conn.Read(hsBuf[:])
		if err != nil {
			// The Read() could have returned Data and an error, but there is
			// no point in continuing on an EOF or whatever.
			return err
		}
		conn.ReceiveBuffer.Write(hsBuf[:n])

		seed, err := hs.parseClientHandshake(sf.replayFilter, conn.ReceiveBuffer.Bytes())
		if err == ErrMarkNotFoundYet {
			continue
		} else if err != nil {
			return err
		}
		conn.ReceiveBuffer.Reset()

		if err := conn.Conn.SetDeadline(time.Time{}); err != nil {
			return nil
		}

		// Use the derived key material to intialize the link crypto.
		okm := ntor.Kdf(seed, framing.KeyLength*2)
		conn.Encoder = framing.NewEncoder(okm[framing.KeyLength:])
		conn.Decoder = framing.NewDecoder(okm[:framing.KeyLength])

		break
	}

	// Since the current and only implementation always sends a PRNG seed for
	// the length obfuscation, this makes the amount of Data received from the
	// server inconsistent with the length sent from the client.
	//
	// Rebalance this by tweaking the client mimimum padding/server maximum
	// padding, and sending the PRNG seed unpadded (As in, treat the PRNG seed
	// as part of the server response).  See inlineSeedFrameLength in
	// handshake_ntor.go.

	// Generate/send the response.
	blob, err := hs.generateHandshake()
	if err != nil {
		return err
	}
	var frameBuf bytes.Buffer
	if _, err = frameBuf.Write(blob); err != nil {
		return err
	}

	// Send the PRNG seed as the first packet.
	if err := conn.MakePacket(&frameBuf, PacketTypePrngSeed, sf.lenSeed.Bytes()[:], uint16(MaxPacketPayloadLength-len(sf.lenSeed.Bytes()[:]))); err != nil {
		return err
	}
	if _, err = conn.Conn.Write(frameBuf.Bytes()); err != nil {
		return err
	}

	return nil
}

type ReadPacketsFuncType func() error

// utils
// Read interface with customized readPackets function
func (conn *DefConn) MyRead(b []byte, readPacketsFunc ReadPacketsFuncType) (n int, err error) {
	// If there is no payload from the previous Read() calls, consume Data off
	// the network.  Not all Data received is guaranteed to be usable payload,
	// so do this in a loop till Data is present or an error occurs.
	for conn.ReceiveDecodedBuffer.Len() == 0 {
		err = readPacketsFunc()
		if err == framing.ErrAgain {
			// Don't proagate this back up the call stack if we happen to break
			// out of the loop.
			err = nil
			continue
		} else if err != nil {
			break
		}
	}

	// Even if err is set, attempt to do the read anyway so that all decoded
	// Data gets relayed before the connection is torn down.
	if conn.ReceiveDecodedBuffer.Len() > 0 {
		var berr error
		n, berr = conn.ReceiveDecodedBuffer.Read(b)
		if err == nil {
			// Only propagate berr if there are not more important (fatal)
			// errors from the network/crypto/packet processing.
			err = berr
		}
	}
	return
}

func (conn *DefConn) Read(b []byte) (n int, err error) {
	// If there is no payload from the previous Read() calls, consume Data off
	// the network.  Not all Data received is guaranteed to be usable payload,
	// so do this in a loop till Data is present or an error occurs.
	for conn.ReceiveDecodedBuffer.Len() == 0 {
		err = conn.ReadPackets()
		if err == framing.ErrAgain {
			// Don't proagate this back up the call stack if we happen to break
			// out of the loop.
			err = nil
			continue
		} else if err != nil {
			break
		}
	}

	// Even if err is set, attempt to do the read anyway so that all decoded
	// Data gets relayed before the connection is torn down.
	if conn.ReceiveDecodedBuffer.Len() > 0 {
		var berr error
		n, berr = conn.ReceiveDecodedBuffer.Read(b)
		if err == nil {
			// Only propagate berr if there are not more important (fatal)
			// errors from the network/crypto/packet processing.
			err = berr
		}
	}
	return
}

// utils
func (conn *DefConn) Send() {
	//A dedicated function responsible for sending out packets coming from conn.SendChan
	//Err is propagated via conn.ErrChan
	for {
		select {
		case _, ok := <-conn.CloseChan:
			if !ok {
				log.Infof("[Routine] Send routine exits by closedChan.")
				return
			}
		case packetInfo := <-conn.SendChan:
			pktType := packetInfo.PktType
			data := packetInfo.Data
			padLen := packetInfo.PadLen
			var frameBuf bytes.Buffer
			err := conn.MakePacket(&frameBuf, pktType, data, padLen)
			if err != nil {
				conn.ErrChan <- err
				log.Infof("[Routine] Send routine exits by make pkt err.")
				return
			}
			_, wtErr := conn.Conn.Write(frameBuf.Bytes())
			if wtErr != nil {
				conn.ErrChan <- wtErr
				log.Infof("[Routine] Send routine exits by write err.")
				return
			}

			if !conn.IsServer && LogEnabled {
				log.Debugf("[TRACE] Scd,%v,%-3d+ %d,%v,%v", pktType, len(data), padLen, time.Now().UnixMicro(), conn.Conn.LocalAddr().String()) // Scd: Send client defconn
			} else {
				//log.Noticef("Sgd,%v,%-3d+ %d,%v,%v", pktType, len(data), padLen, time.Now().UnixMicro(), conn.Conn.RemoteAddr().String()) // Sgd: Send guard defconn
			}
		}
	}
}

func (conn *DefConn) ReadFrom(r io.Reader) (written int64, err error) {
	log.Infof("[State] Enter parent copyloop state: %v (%v is stateStart, %v is statStop)", conn.ConnState.LoadCurState(), StateStart, StateStop)
	defer close(conn.CloseChan)

	var receiveBuf utils.SafeBuffer //read payload from upstream and buffer here

	//create a go routine to send out packets to the wire
	go conn.Send()

	// this go routine regularly check the real throughput
	// if it is small, change to stop state
	go func() {
		ticker := time.NewTicker(TWindow)
		defer ticker.Stop()
		for {
			select {
			case _, ok := <-conn.CloseChan:
				if !ok {
					log.Infof("[Routine] Ticker routine exits by closeChan.")
					return
				}
			case <-ticker.C:
				log.Debugf("[State] Real Sent: %v, Real Receive: %v, curState: %s at %v.", conn.NRealSegSentLoad(), conn.NRealSegRcvLoad(), StateMap[conn.ConnState.LoadCurState()], time.Now().Format("15:04:05.000000"))
				if !conn.IsServer && conn.ConnState.LoadCurState() != StateStop && (conn.NRealSegSentLoad() < 2 || conn.NRealSegRcvLoad() < 2) {
					log.Infof("[State] %s -> %s.", StateMap[conn.ConnState.LoadCurState()], StateMap[StateStop])
					conn.ConnState.SetState(StateStop)
					conn.SendChan <- PacketInfo{PktType: PacketTypeSignalStop, Data: []byte{}, PadLen: MaxPacketPaddingLength}
				}
				conn.NRealSegReset() //reset counter
			}
		}
	}()

	for {
		select {
		case conErr := <-conn.ErrChan:
			log.Infof("downstream copy loop terminated at %v. Reason: %v", time.Now().Format("15:04:05.000000"), conErr)
			return written, conErr
		default:
			buf := make([]byte, 65535)
			rdLen, err := r.Read(buf[:])
			if err != nil {
				log.Errorf("Exit by read err:%v", err)
				return written, err
			}
			if rdLen > 0 {
				receiveBuf.Write(buf[:rdLen])
			} else {
				log.Errorf("BUG? read 0 bytes, err: %v", err)
				return written, io.EOF
			}
			//signal server to start if there is more than one cell coming
			// else switch to padding state
			// stop -> ready -> start
			if !conn.IsServer {
				if (conn.ConnState.LoadCurState() == StateStop && rdLen > MaxPacketPayloadLength) ||
					(conn.ConnState.LoadCurState() == StateReady) {
					log.Infof("[State] %s -> %s.", StateMap[conn.ConnState.LoadCurState()], StateMap[StateStart])
					conn.ConnState.SetState(StateStart)
					conn.SendChan <- PacketInfo{PktType: PacketTypeSignalStart, Data: []byte{}, PadLen: MaxPacketPaddingLength}
				} else if conn.ConnState.LoadCurState() == StateStop {
					log.Infof("[State] %s -> %s.", StateMap[StateStop], StateMap[StateReady])
					conn.ConnState.SetState(StateReady)
				}
			}
			for receiveBuf.GetLen() > 0 {
				var payload [MaxPacketPayloadLength]byte
				rdLen, rdErr := receiveBuf.Read(payload[:])
				written += int64(rdLen)
				if rdErr != nil {
					log.Infof("Exit by read buffer err:%v", rdErr)
					return written, rdErr
				}
				conn.SendChan <- PacketInfo{PktType: PacketTypePayload, Data: payload[:rdLen], PadLen: uint16(MaxPacketPaddingLength - rdLen)}
				conn.NRealSegSentIncrement()
			}
		}
	}
}

func (conn *DefConn) SetDeadline(t time.Time) error {
	return syscall.ENOTSUP
}

func (conn *DefConn) SetWriteDeadline(t time.Time) error {
	return syscall.ENOTSUP
}

func (conn *DefConn) closeAfterDelay(sf *DefConnServerFactory, startTime time.Time) {
	// I-it's not like I w-wanna handshake with you or anything.  B-b-baka!
	defer conn.Conn.Close()

	delay := time.Duration(sf.closeDelay)*time.Second + serverHandshakeTimeout
	deadline := startTime.Add(delay)
	if time.Now().After(deadline) {
		return
	}

	if err := conn.Conn.SetReadDeadline(deadline); err != nil {
		return
	}

	// Consume and discard Data on this connection until the specified interval
	// passes.
	_, _ = io.Copy(ioutil.Discard, conn.Conn)
}

var _ base.ClientFactory = (*DefConnClientFactory)(nil)
var _ base.ServerFactory = (*DefConnServerFactory)(nil)
var _ base.Transport = (*Transport)(nil)
var _ net.Conn = (*DefConn)(nil)
