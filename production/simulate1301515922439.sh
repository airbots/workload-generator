nohup hadoop jar /opt/hadoop/hadoop-0.20.2-test.jar loadgen -m 1 -keepmap 100 -keepred 200 -indir 	/user/generator/file58/file58 -outdir /user/generator/out1301515922439
 mv nohup.out results1301515922439
date +%s > finishTime_1301515922439.txt