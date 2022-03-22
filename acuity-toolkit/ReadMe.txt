Conversion_scripts:
1、Convert Model to case code:
    you only need Configure corresponding parameters and execute:
        sh 0_import_model.sh
	sh 1_quantize_model.sh
	sh 2_export_case_code.sh
   
   Note: hybrid case
        in dir demo_hybrid:
	    1.Execute: sh 0_import_model.sh
            2.Execute: sh 1_quantize_model.sh
	    3.modify the xxxx.quantize file
	    4.Execute: sh hybrid_1_quantize_model.sh
	    5.Execute: sh hybrid_2_export_case_code.sh

2、Get the mapping of the output
    python extractoutput.py   xxx.json

	


