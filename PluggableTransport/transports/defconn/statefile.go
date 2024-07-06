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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"git.torproject.org/pluggable-transports/goptlib.git"
	"PluggableTransport/common/csrand"
	"PluggableTransport/common/drbg"
	"PluggableTransport/common/ntor"
	"io/ioutil"
	"os"
	"path"
	"strings"
)

const (
	StateFile  = "defconn_state.json"
	BridgeFile = "defconn_bridgeline.txt"

	certSuffix = "=="
	certLength = ntor.NodeIDLength + ntor.PublicKeyLength
)

type JsonServerState struct {
	NodeID     string         `json:"node-id"`
	PrivateKey string         `json:"private-key"`
	PublicKey  string         `json:"public-key"`
	DrbgSeed   string         `json:"drbg-seed"`
}

type JsonServerStateInterface interface {
	SetNodeId(nodeID string)
	GetNodeId() string
	SetPublicKey(publicKey string)
	GetPublicKey() string
	SetPrivateKey(privateKey string)
	GetPrivateKey() string
	SetDrbg(drbgSeed string)
	GetDrbg() string
}

func (js *JsonServerState) SetNodeId(nodeID string) {
	js.NodeID = nodeID
}

func (js *JsonServerState) GetNodeId() string {
	return js.NodeID
}

func (js *JsonServerState) SetPublicKey(publicKey string) {
	js.PublicKey = publicKey
}

func (js *JsonServerState) GetPublicKey() string {
	return js.PublicKey
}

func (js *JsonServerState) SetPrivateKey(privateKey string) {
	js.PrivateKey = privateKey
}

func (js *JsonServerState) GetPrivateKey() string {
	return js.PrivateKey
}

func (js *JsonServerState) SetDrbg(drbgSeed string) {
	js.DrbgSeed = drbgSeed
}

func (js *JsonServerState) GetDrbg() string {
	return js.DrbgSeed
}

type defConnServerCert struct {
	raw []byte
}

func (cert *defConnServerCert) String() string {
	return strings.TrimSuffix(base64.StdEncoding.EncodeToString(cert.raw), certSuffix)
}

func (cert *defConnServerCert) unpack() (*ntor.NodeID, *ntor.PublicKey) {
	if len(cert.raw) != certLength {
		panic(fmt.Sprintf("cert length %d is invalid", len(cert.raw)))
	}

	nodeID, _ := ntor.NewNodeID(cert.raw[:ntor.NodeIDLength])
	pubKey, _ := ntor.NewPublicKey(cert.raw[ntor.NodeIDLength:])

	return nodeID, pubKey
}

func serverCertFromString(encoded string) (*defConnServerCert, error) {
	decoded, err := base64.StdEncoding.DecodeString(encoded + certSuffix)
	if err != nil {
		return nil, fmt.Errorf("failed to decode cert: %s", err)
	}

	if len(decoded) != certLength {
		return nil, fmt.Errorf("cert length %d is invalid", len(decoded))
	}

	return &defConnServerCert{raw: decoded}, nil
}

func serverCertFromState(st *DefConnServerState) *defConnServerCert {
	cert := new(defConnServerCert)
	cert.raw = append(st.nodeID.Bytes()[:], st.identityKey.Public().Bytes()[:]...)
	return cert
}

type DefConnServerState struct {
	nodeID      *ntor.NodeID
	identityKey *ntor.Keypair
	drbgSeed    *drbg.Seed
	cert        *defConnServerCert
}

func (st *DefConnServerState) ClientString() string {
	return fmt.Sprintf("%s=%s ",
		certArg, st.cert)
}

func ServerStateFromArgs(stateDir string, args *pt.Args) (*DefConnServerState, error) {
	js, err := ServerStateFromArgsInternal(stateDir, StateFile, args)
	if err != nil {
		return nil, err
	}
	return ServerStateFromJsonServerState(stateDir, StateFile, &js)
}

func ServerStateFromArgsInternal(stateDir string, stateFileName string, args *pt.Args) (JsonServerState, error) {
	var nodeIDOk, privKeyOk, seedOk bool
	var js JsonServerState
	js.NodeID, nodeIDOk = args.Get(nodeIDArg)
	js.PrivateKey, privKeyOk = args.Get(privateKeyArg)
	js.DrbgSeed, seedOk = args.Get(seedArg)

	// Either a private key, node id, and seed are ALL specified, or
	// they should be loaded from the state file.
	if !privKeyOk && !nodeIDOk && !seedOk {
		if err := jsonServerStateFromFile(stateDir, stateFileName, &js); err != nil {
			return JsonServerState{}, err
		}
	} else if !privKeyOk {
		return JsonServerState{}, fmt.Errorf("missing argument '%s'", privateKeyArg)
	} else if !nodeIDOk {
		return JsonServerState{}, fmt.Errorf("missing argument '%s'", nodeIDArg)
	} else if !seedOk {
		return JsonServerState{}, fmt.Errorf("missing argument '%s'", seedArg)
	}
	return js, nil
}

func ServerStateFromJsonServerState(stateDir string, stateFileName string, js JsonServerStateInterface) (*DefConnServerState, error) {

	st, err := ServerStateFromJsonServerStateInternal(js)
	if err != nil {
		return nil, err
	}
	// Generate a human readable summary of the configured endpoint.
	if err = NewBridgeFile(stateDir, BridgeFile, st.ClientString()); err != nil {
		return nil, err
	}
	// Write back the possibly updated server state.
	return &st, WriteJSONServerState(stateDir, stateFileName, js)
}

func ServerStateFromJsonServerStateInternal(js JsonServerStateInterface) (DefConnServerState, error){
	var err error

	var st DefConnServerState
	if st.nodeID, err = ntor.NodeIDFromHex(js.GetNodeId()); err != nil {
		return DefConnServerState{}, err
	}
	if st.identityKey, err = ntor.KeypairFromHex(js.GetPrivateKey()); err != nil {
		return DefConnServerState{}, err
	}
	if st.drbgSeed, err = drbg.SeedFromHex(js.GetDrbg()); err != nil {
		return DefConnServerState{}, err
	}

	st.cert = serverCertFromState(&st)
	return st, err
}


func jsonServerStateFromFile(stateDir string, stateFileName string, js JsonServerStateInterface) error {
	fPath := path.Join(stateDir, stateFileName)
	f, err := ioutil.ReadFile(fPath)
	if err != nil {
		if os.IsNotExist(err) {
			if err = NewJsonServerState(stateDir, stateFileName, js); err == nil {
				return nil
			}
		}
		return err
	}

	if err := json.Unmarshal(f, js); err != nil {
		return fmt.Errorf("failed to load statefile '%s': %s", fPath, err)
	}

	return nil
}

func NewJsonServerState(stateDir string, stateFileName string, js JsonServerStateInterface) (err error) {
	// Generate everything a server needs, using the cryptographic PRNG.
	var st DefConnServerState
	rawID := make([]byte, ntor.NodeIDLength)
	if err = csrand.Bytes(rawID); err != nil {
		return
	}
	if st.nodeID, err = ntor.NewNodeID(rawID); err != nil {
		return
	}
	if st.identityKey, err = ntor.NewKeypair(false); err != nil {
		return
	}
	if st.drbgSeed, err = drbg.NewSeed(); err != nil {
		return
	}

	// Encode it into JSON format and write the state file.
	js.SetNodeId(st.nodeID.Hex())
	js.SetPrivateKey(st.identityKey.Private().Hex())
	js.SetPublicKey(st.identityKey.Public().Hex())
	js.SetDrbg(st.drbgSeed.Hex())

	return WriteJSONServerState(stateDir, stateFileName, js)
}

func WriteJSONServerState(stateDir string, stateFileName string, js JsonServerStateInterface) error {
	var err error
	var encoded []byte
	if encoded, err = json.Marshal(js); err != nil {
		return err
	}
	if err = ioutil.WriteFile(path.Join(stateDir, stateFileName), encoded, 0600); err != nil {
		return err
	}

	return nil
}

func NewBridgeFile(stateDir string, bridgeFileName string, bridgeLineStr string) error {
	const prefix = "# DefConn torrc client bridge line\n" +
		"#\n" +
		"# This file is an automatically generated bridge line based on\n" +
		"# the current defConnproxy configuration.  EDITING IT WILL HAVE\n" +
		"# NO EFFECT.\n" +
		"#\n" +
		"# Before distributing this Bridge, edit the placeholder fields\n" +
		"# to contain the actual values:\n" +
		"#  <IP ADDRESS>  - The public IP address of your DefConn bridge.\n" +
		"#  <PORT>        - The TCP/IP port of your DefConn bridge.\n" +
		"#  <FINGERPRINT> - The bridge's fingerprint.\n\n"

	bridgeLine := fmt.Sprintf("Bridge DefConn <IP ADDRESS>:<PORT> <FINGERPRINT> %s\n",
		bridgeLineStr)

	tmp := []byte(prefix + bridgeLine)
	if err := ioutil.WriteFile(path.Join(stateDir, bridgeFileName), tmp, 0600); err != nil {
		return err
	}

	return nil
}
