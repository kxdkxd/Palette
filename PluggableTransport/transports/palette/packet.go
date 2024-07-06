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

package palette

import (
	"encoding/binary"
	"fmt"
	"PluggableTransport/common/drbg"
	"PluggableTransport/common/log"
	"PluggableTransport/transports/defconn"
	"PluggableTransport/transports/defconn/framing"
	"math/rand"
	"sync/atomic"
	"time"
)

func (conn *paletteConn) readPackets() (err error) {
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
		} else if decLen < defconn.PacketOverhead {
			err = defconn.InvalidPacketLengthError(decLen)
			break
		}

		// Decode the packet.
		pkt := decoded[0:decLen]
		pktType := pkt[0]
		payloadLen := binary.BigEndian.Uint16(pkt[1:])
		if int(payloadLen) > len(pkt)-defconn.PacketOverhead {
			err = defconn.InvalidPayloadLengthError(int(payloadLen))
			break
		}
		payload := pkt[3 : 3+payloadLen]

		if !conn.IsServer && pktType != defconn.PacketTypePrngSeed && defconn.LogEnabled {
			log.Debugf("[TRACE] Rcp,%v,%-3d+ %d,%v,%v", pktType, -int64(payloadLen), -(int64(decLen-defconn.PacketOverhead) - int64(payloadLen)), time.Now().UnixMicro(), conn.Conn.LocalAddr().String())
		} else {
			// placeholder
		}

		switch pktType {
		case defconn.PacketTypePayload:
			if payloadLen > 0 {
				conn.ReceiveDecodedBuffer.Write(payload)
				conn.NRealSegRcvIncrement()
			}
		case defconn.PacketTypePrngSeed:
			// Only regenerate the distribution if we are the client.
			if len(payload) == defconn.SeedPacketPayloadLength && !conn.IsServer {
				var seed *drbg.Seed
				seed, err = drbg.SeedFromBytes(payload)
				if err != nil {
					break
				}
				conn.LenDist.Reset(seed)
			}
		case defconn.PacketTypeSignalStart:
			// a signal from client to make server change to stateStart
			if !conn.IsServer {
				panic(fmt.Sprintf("Client receive SignalStart pkt from server? "))
			}
			if conn.ConnState.LoadCurState() != defconn.StateStart {
				log.Debugf("[State] Client signal: %s -> %s.", defconn.StateMap[conn.ConnState.LoadCurState()], defconn.StateMap[defconn.StateStart])
				conn.flagEndDownload = false
				conn.flagFirstDownload = false
				conn.totalPacketCountReset()
				conn.thresholdDown = rand.Intn(conn.U_download) + 1
				conn.seqId = int(payload[0])
				conn.nowPrunedCenter, _, conn.downloadKeep = conn.getStrategy(downloadDirection, conn.seqId)
				atomic.StoreInt32(&conn.resDown, 0)
				atomic.StoreInt32(&conn.nowIdx, 0)
				conn.ConnState.SetState(defconn.StateStart)
			}
		case defconn.PacketTypeSignalStop:
			// a signal from client to make server change to stateStop
			if !conn.IsServer { // if is client, exit, rest all for server
				panic(fmt.Sprintf("Client receive SignalStop pkt from server? "))
			}
			if conn.ConnState.LoadCurState() != defconn.StateStop {
				log.Debugf("[State] Client signal: %s -> %s.", defconn.StateMap[conn.ConnState.LoadCurState()], defconn.StateMap[defconn.StateStop])
				conn.ConnState.SetState(defconn.StateStop)
			}
		case defconn.PacketTypeDummy:
		default:
			// Ignore unknown packet types.
		}
	}

	// Read errors (all fatal) take priority over various frame processing errors.
	if rdErr != nil {
		return rdErr
	}

	return
}
