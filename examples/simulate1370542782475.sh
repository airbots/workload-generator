nohup time hadoop jar wordCount.jar wordCount -dl 200 -input 	/user/generator/file52/file52 	/user/generator/file9/file9 	/user/generator/file97/file97 	/user/generator/file34/file34 	/user/generator/file76/file76 	/user/generator/file14/file14 	/user/generator/file68/file68 	/user/generator/file56/file56 	/user/generator/file6/file6 	/user/generator/file62/file62 	/user/generator/file81/file81 	/user/generator/file74/file74 	/user/generator/file12/file12 	/user/generator/file90/file90 	/user/generator/file7/file7 	/user/generator/file97/file97 	/user/generator/file47/file47 	/user/generator/file72/file72 	/user/generator/file2/file2 	/user/generator/file47/file47 	/user/generator/file92/file92 	/user/generator/file95/file95 	/user/generator/file50/file50 -output /user/generator/out1370542782475
 mv nohup.out results1370542782475
hadoop fs -rmr /user/generator/out1370542782475 
date +%s > finishTime_1370542782475.txt