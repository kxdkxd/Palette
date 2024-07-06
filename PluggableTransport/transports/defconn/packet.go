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

package defconn

import (
	"encoding/binary"
	"fmt"
	"PluggableTransport/common/log"
	"io"
	"time"

	"PluggableTransport/common/drbg"
	"PluggableTransport/transports/defconn/framing"
)

const (
	PacketOverhead = 2 + 1
	//MaxPacketPayloadLength  = framing.MaximumFramePayloadLength - PacketOverhead
	MaxPacketPayloadLength  = 536 //size of a Tor cell wrapped by TLS
	MaxPacketPaddingLength  = MaxPacketPayloadLength
	SeedPacketPayloadLength = seedLength

	ConsumeReadSize = framing.MaximumSegmentLength * 16
)

const (
	PacketTypePayload = iota
	PacketTypeDummy
	PacketTypePrngSeed
	PacketTypeSignalStart
	PacketTypeSignalStop
	PacketTypeFinish
)

var PktTypeMap = map[uint8]string{
	PacketTypePayload:     "Payload",
	PacketTypeDummy:       "Dummy",
	PacketTypePrngSeed:    "PrngSeed",
	PacketTypeSignalStart: "SigStart",
	PacketTypeSignalStop:  "SigStop",
	PacketTypeFinish:      "SigFinish",
}

// InvalidPacketLengthError is the error returned when decodePacket detects a
// invalid packet length/
type InvalidPacketLengthError int

func (e InvalidPacketLengthError) Error() string {
	return fmt.Sprintf("packet: Invalid packet length: %d", int(e))
}

// InvalidPayloadLengthError is the error returned when decodePacket rejects the
// payload length.
type InvalidPayloadLengthError int

type PacketInfo struct {
	PktType uint8
	Data    []byte
	PadLen  uint16
}

func (e InvalidPayloadLengthError) Error() string {
	return fmt.Sprintf("packet: Invalid payload length: %d", int(e))
}

var zeroPadBytes [MaxPacketPaddingLength]byte

func (conn *DefConn) MakePacket(w io.Writer, pktType uint8, data []byte, padLen uint16) error {
	var pkt [framing.MaximumFramePayloadLength]byte

	if len(data)+int(padLen) > MaxPacketPayloadLength {
		panic(fmt.Sprintf("BUG: MakePacket() len(Data) + PadLen > MaxPacketPayloadLength: %d + %d > %d",
			len(data), padLen, MaxPacketPayloadLength))
	}

	// Packets are:
	//   uint8_t type      PacketTypePayload (0x00)
	//   uint16_t length   Length of the payload (Big Endian).
	//   uint8_t[] payload Data payload.
	//   uint8_t[] padding Padding.
	pkt[0] = pktType
	binary.BigEndian.PutUint16(pkt[1:], uint16(len(data)))
	if len(data) > 0 {
		copy(pkt[3:], data[:])
	}
	copy(pkt[3+len(data):], zeroPadBytes[:padLen])

	pktLen := PacketOverhead + len(data) + int(padLen)

	// Encode the packet in an AEAD frame.
	var frame [framing.MaximumSegmentLength]byte
	frameLen, err := conn.Encoder.Encode(frame[:], pkt[:pktLen])
	if err != nil {
		// All Encoder errors are fatal.
		return err
	}
	wrLen, err := w.Write(frame[:frameLen])
	if err != nil {
		return err
	} else if wrLen < frameLen {
		return io.ErrShortWrite
	}

	return nil
}

func (conn *DefConn) ReadPackets() (err error) {
	// Attempt to read off the network.
	rdLen, rdErr := conn.Conn.Read(conn.ReadBuffer)
	conn.ReceiveBuffer.Write(conn.ReadBuffer[:rdLen])

	var decoded [framing.MaximumFramePayloadLength]byte
	for conn.ReceiveBuffer.Len() > 0 {
		// Decrypt an AEAD frame.
		decLen := 0
		decLen, err = conn.Decoder.Decode(decoded[:], conn.ReceiveBuffer)
		if err == framing.ErrAgain {
			break
		} else if err != nil {
			break
		} else if decLen < PacketOverhead {
			err = InvalidPacketLengthError(decLen)
			break
		}

		// Decode the packet.
		pkt := decoded[0:decLen]
		pktType := pkt[0]
		payloadLen := binary.BigEndian.Uint16(pkt[1:])
		if int(payloadLen) > len(pkt)-PacketOverhead {
			err = InvalidPayloadLengthError(int(payloadLen))
			break
		}
		payload := pkt[3 : 3+payloadLen]

		if !conn.IsServer && pktType != PacketTypePrngSeed && LogEnabled {
			log.Debugf("[TRACE] Rcd,%v,%-3d+ %d,%v,%v", pktType, -int64(payloadLen), -(int64(decLen-PacketOverhead) - int64(payloadLen)), time.Now().UnixMicro(), conn.Conn.LocalAddr().String())
		} else {
			log.Debugf("[Rcv] defconn %-8s, %-3d+ %-3d bytes at %v", PktTypeMap[pktType], -int64(payloadLen), -(int64(decLen-PacketOverhead) - int64(payloadLen)), time.Now().Format("15:04:05.000000000"))
		}

		switch pktType {
		case PacketTypePayload:
			if payloadLen > 0 {
				conn.ReceiveDecodedBuffer.Write(payload)
				conn.NRealSegRcvIncrement()
			}
		case PacketTypePrngSeed:
			// Only regenerate the distribution if we are the client.
			if len(payload) == SeedPacketPayloadLength && !conn.IsServer {
				var seed *drbg.Seed
				seed, err = drbg.SeedFromBytes(payload)
				if err != nil {
					break
				}
				conn.LenDist.Reset(seed)
			}
		case PacketTypeSignalStart:
			// a signal from client to make server change to stateStart
			if !conn.IsServer {
				panic(fmt.Sprintf("Client receive SignalStart pkt from server? "))
			}
			if conn.ConnState.LoadCurState() != StateStart {
				log.Debugf("[State] Client signal: %s -> %s.", StateMap[conn.ConnState.LoadCurState()], StateMap[StateStart])
				conn.ConnState.SetState(StateStart)
			}
		case PacketTypeSignalStop:
			// a signal from client to make server change to stateStop
			if !conn.IsServer {
				panic(fmt.Sprintf("Client receive SignalStop pkt from server? "))
			}
			if conn.ConnState.LoadCurState() != StateStop {
				log.Debugf("[State] Client signal: %s -> %s.", StateMap[conn.ConnState.LoadCurState()], StateMap[StateStop])
				conn.ConnState.SetState(StateStop)
			}
		case PacketTypeFinish:
		case PacketTypeDummy:
		default:
			// Ignore unknown packet types.
		}
	}

	// Read errors (all fatal) take priority over various frame processing
	// errors.
	if rdErr != nil {
		return rdErr
	}

	return
}
