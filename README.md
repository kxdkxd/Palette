# Palette
This is the source code for paper "[Real-Time Website Fingerprinting Defense via Traffic Cluster Anonymization](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a263/1WPcZnZILHa)", accepted in IEEE Symposium on Security and Privacy (S\&P) 2024.

# Disclaimer
:warning: This repository is intended for RESEARCH PURPOSES ONLY!:warning:

# Repo Structure
**Simulation:** Simulation code for Palette.

**PluggableTransport**: Implementation code for Palette based on WFDefProxy[1] and obfs4proxy[2]. The main logic of Palette is at *PluggableTransport/transports/palette*

# Setup For Simulation
## Environment
* Python Version: 3.10  

Use this command to install the required packages.
```commandline
pip3 install -r requirements.txt
```

## Datasets
We sincerely thank the authors for sharing their dataset.
The public real-world dataset used in our experiments is listed below:
* [Deep Fingerprinting](https://dl.acm.org/doi/pdf/10.1145/3243734.3243768): It contains 95 websites, each with 1,000 undefended traces, for closed-world evaluation. Excluding the 95 websites, it also includes 40,000 websites for open-world evaluation, each with only 1 undefended trace. This dataset is provided by [Rahman et al.](https://github.com/msrocean/Tik_Tok), and you can find the dataset on the [google drive link](https://drive.google.com/drive/folders/1k6X8PjKTXNalCiUQudx-HyqoAXVXRknL).

## Extract TAM Feature
We use the code from Robust Fingerprinting to extract the Traffic Aggregation Matrix of traces.
The following command can extract a randomly split training, validation, and testing dataset.
```commandline
python extract-list.py
```

You can set the parameters in const.py. 

## Anonymity Set Generation
The following command can run website clustering to generate anonymity sets and super-matrices
```
python cluster.py
```

You can set the parameters in const.py.

## Super-Matrix Refinement
Run following command to shrink the super-matrices
```
python refinement.py
```

You can set the parameters in parse_args(). 

## Regularization
Run following command to regulate the undefended traces.
```
python regularization.py
```

You can set the parameters in parse_args(). 

# Setup For PluggableTransport
## Build
Our go version is go1.18.1 linux/amd64
```bash
go build -o obfs4proxy/obfs4proxy ./obfs4proxy
```
Then copy the compiled obfs4proxy file to the path defined in torrc.

## Run Palette
### Bridge
The torrc file of Tor Bridge:
```
DataDirectory /home/example/Browser/TorBrowser/Tor/palette-data/bridge
Log notice stdout
SOCKSPort 9052
AssumeReachable 1
PublishServerDescriptor 0
Exitpolicy reject *:*
ORPort auto
ExtORPort auto
Nickname "palette"
BridgeRelay 1
ServerTransportListenAddr palette 0.0.0.0:8089
ServerTransportPlugin palette exec /home/example/Browser/TorBrowser/Tor/obfs4proxy
ServerTransportOptions palette U_upload=15 U_download=15 B=1 Alpha_upload=0.25 Alpha_download=0.25
```
The Palette defense parameter is defined above.

Three additional files for Palette to generate the trace are needed: `prunedCenterSet.json`, `PMF_upload.json`, `PMF_download.json`. Each of them is an array. You can generate these 3 files with the code inside this repo.

If tor boots up, a file named `defconn_bridgeline.txt` will be generated in `/home/example/Browser/TorBrowser/Tor/palette-data/bridge/pt_state`, containing a `cert` parameter that needs to be copied into client's torrc for handshaking.

### Client
The torrc file of Tor Client is as follows. Modify the `[BRIDGE_IP]` to your own bridge IP address.
```
DataDirectory /home/example/Browser/TorBrowser/Tor/palette-data/client
Log notice stdout    
SOCKSPort 9050
TransPort 0.0.0.0:9040
ControlPort 9051  
CookieAuthentication 0
HashedControlPassword ""
UseBridges 1    
ClientTransportPlugin palette exec /home/example/Browser/TorBrowser/Tor/obfs4proxy
Bridge palette [BRIDGE_IP]:8089 cert=lok6+guthU2u4IsLkj+fdXK6tNgVxbyq2Ab7fOljVZ2NXkmpIgBxyKH/1kbh03ypNSoeXg U_upload=15 U_download=15 B=1 Alpha_upload=0.25 Alpha_download=0.25
```

The additional three files are also needed. The Pluggable Transport will bind the local port 7999. When the client and the bridge are successfully connected, a http rpc is needed to send a command to the client to start the defense, with the current site's  cluster ID in the parameter. For example: `requests.get('http://localhost:7999/enter', params={'seqId': seq_id_dict[url]})`

# References
[1] J. Gong, et al. "WFDefProxy:Modularly Implementing and Empirically Evaluating Website Fingerprinting Defenses"

[2] Yawning, Angel. "obfs4 - The obfourscator"

# Acknowledgements
We express our gratitude to our reviewers and shepherds for their insightful suggestions and comments. We also extend our thanks to Jiajun Gong for generously sharing the code for WfDefProxy and WFCrawler.
# Citation
If you find this work useful for your research, please cite our paper using the following BibTeX entry.
```BibTeX
@inproceedings{shen2024real,
    author = {M. Shen and K. Ji and J. Wu and Q. Li and X. Kong and K. Xu and L. Zhu},
    booktitle = {2024 IEEE Symposium on Security and Privacy (SP)},
    title = {Real-Time Website Fingerprinting Defense via Traffic Cluster Anonymization},
    pages={263--263},
    year = {2024},
    month = {may},
    publisher = {IEEE Computer Society}
}
```

# Contact
If you have any questions, please get in touch with us.
* Prof. Meng Shen ([shenmeng@bit.edu.cn](shenmeng@bit.edu.cn))
* Kexin Ji ([jikexin@bit.edu.cn](jikexin@bit.edu.cn))
* Jinhe Wu ([jinhewu@bit.edu.cn](jinhewu@bit.edu.cn))
* Xiangdong Kong ([xiangdongkong@bit.edu.cn](xiangdongkong@bit.edu.cn))

More detailed information about the research of Meng Shen Lab can be found [here](https://mengshen-office.github.io/).
