Prepare envirment:
	Install Python3 packages: (note: need python versoin 3.5.2)
		sudo apt-get install python3.5 python3-pip python3-virtualenv

	Install tensorflow and other packages for pip:
		for req in $(cat requirements.txt); do pip3 install $req; done

Note: if network speed is slow and the pip installation is affected, you can execute:
    for req in $(cat requirements.txt); do pip3 install $req -i https://pypi.tuna.tsinghua.edu.cn/simple; done

	if (ReadTimeoutError: HTTPSConnectionPool(host='pypi.tuna.tsinghua.edu.cn', port=443): Read timed out) appear ?
		you can wget the whl package ,and then execute cmd: pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple  xxxxxx.whl
		get whl method: (use torch for example)
		1、check the log after you execute:  pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.2.0
			/********log************/
			root@30950dc82678:/home# pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.2.0
			WARNING: pip is being invoked by an old script wrapper. This will fail in a future version of pip.
			Please see https://github.com/pypa/pip/issues/5599 for advice on fixing the underlying issue.
			To avoid this problem you can invoke Python with '-m pip' instead of running pip directly.
			Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
			Collecting torch==1.2.0
			  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/30/57/d5cceb0799c06733eefce80c395459f28970ebb9e896846ce96ab579a3f1/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl (748.8 MB)
				 |                                | 12.9 MB 2.1 MB/s eta 0:05:58ERROR: Exception:
			/********log************/

		2、find the whl package download path after the  Downloading Keyword
		3、and then  you can download the package: wget  https://pypi.tuna.tsinghua.edu.cn/packages/30/57/d5cceb0799c06733eefce80c395459f28970ebb9e896846ce96ab579a3f1/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
		4、pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple  torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
	

Conversion_scripts:
1、Convert Model to case code:
	you only need Configure corresponding parameters and execute:
		sh 0_import_model.sh
		sh 1_quantize_model.sh
		sh 2_export_case_code.sh


2、Get the mapping of the output
	python extractoutput.py   xxx.json

	


