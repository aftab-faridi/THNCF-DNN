all:
	${HOME}/MATLAB/R2024a/bin/matlab -nodisplay -nosplash -nodesktop -r "try, data, catch e, disp(getReport(e)), end, exit" > matlab.log 2>&1
	python nu_cf.py > model.log
