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
	"fmt"
	"git.torproject.org/pluggable-transports/goptlib.git"
	"PluggableTransport/transports/defconn"
	"strconv"
)

type jsonServerState struct {
	defconn.JsonServerState

	U_upload       int     `json:"U_upload"`
	U_download     int     `json:"U_download"`
	B              int     `json:"B"`
	Alpha_upload   float32 `json:"Alpha_upload"`
	Alpha_download float32 `json:"Alpha_download"`
}

type paletteServerState struct {
	defconn.DefConnServerState

	U_upload       int
	U_download     int
	B              int
	Alpha_upload   float32
	Alpha_download float32
}

func (st *paletteServerState) clientString() string {
	return st.DefConnServerState.ClientString() +
		fmt.Sprintf("%s=%d %s=%d %s=%d %s=%f %s=%f",
			U_uploadArg, st.U_upload, U_downloadArg, st.U_download, BArg, st.B, Alpha_uploadArg, st.Alpha_upload, Alpha_downloadArg, st.Alpha_download)
}

func serverStateFromArgs(stateDir string, args *pt.Args) (*paletteServerState, error) {
	js, err := defconn.ServerStateFromArgsInternal(stateDir, defconn.StateFile, args)
	if err != nil {
		return nil, err
	}

	U_uploadStr, U_uploadOk := args.Get("U_upload")
	U_downloadStr, U_downloadOk := args.Get("U_download")
	BStr, BOk := args.Get("B")
	Alpha_uploadStr, Alpha_uploadOk := args.Get("Alpha_upload")
	Alpha_downloadStr, Alpha_downloadOk := args.Get("Alpha_download")

	var jsPALETTE jsonServerState
	jsPALETTE.JsonServerState = js

	// The palette params should be independently configurable.

	if U_uploadOk {
		U_upload, err := strconv.Atoi(U_uploadStr)
		if err != nil {
			return nil, fmt.Errorf("malformed '%s'", U_uploadStr)
		}
		jsPALETTE.U_upload = U_upload
	} else {
		return nil, fmt.Errorf("missing argument '%s'", U_uploadStr)
	}
	if U_downloadOk {
		U_download, err := strconv.Atoi(U_downloadStr)
		if err != nil {
			return nil, fmt.Errorf("malformed '%s'", U_downloadStr)
		}
		jsPALETTE.U_download = U_download
	} else {
		return nil, fmt.Errorf("missing argument '%s'", U_downloadStr)
	}
	if BOk {
		B, err := strconv.Atoi(BStr)
		if err != nil {
			return nil, fmt.Errorf("malformed '%s'", BStr)
		}
		jsPALETTE.B = B
	} else {
		return nil, fmt.Errorf("missing argument '%s'", BStr)
	}
	if Alpha_uploadOk {
		Alpha_upload, err := strconv.ParseFloat(Alpha_uploadStr, 32)
		if err != nil {
			return nil, fmt.Errorf("malformed '%s'", Alpha_uploadStr)
		}
		jsPALETTE.Alpha_upload = float32(Alpha_upload)
	} else {
		return nil, fmt.Errorf("missing argument '%s'", Alpha_uploadStr)
	}
	if Alpha_downloadOk {
		Alpha_download, err := strconv.ParseFloat(Alpha_downloadStr, 32)
		if err != nil {
			return nil, fmt.Errorf("malformed '%s'", Alpha_downloadStr)
		}
		jsPALETTE.Alpha_download = float32(Alpha_download)
	} else {
		return nil, fmt.Errorf("missing argument '%s'", Alpha_downloadStr)
	}
	return serverStateFromJSONServerState(stateDir, &jsPALETTE)
}

func serverStateFromJSONServerState(stateDir string, js *jsonServerState) (*paletteServerState, error) {
	st, err := defconn.ServerStateFromJsonServerStateInternal(js)

	var stPALETTE paletteServerState

	stPALETTE.DefConnServerState = st

	stPALETTE.U_upload = js.U_upload
	stPALETTE.U_download = js.U_download
	stPALETTE.B = js.B
	stPALETTE.Alpha_upload = js.Alpha_upload
	stPALETTE.Alpha_download = js.Alpha_download

	// Generate a human readable summary of the configured endpoint.
	if err = defconn.NewBridgeFile(stateDir, defconn.BridgeFile, stPALETTE.clientString()); err != nil {
		return nil, err
	}

	// Write back the possibly updated server state.
	return &stPALETTE, defconn.WriteJSONServerState(stateDir, defconn.StateFile, js)
}
