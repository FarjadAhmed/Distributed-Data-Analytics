#!/bin/sh
for maps in {1,2,4,8,10}
do
    for tasks in {1,2,4,8,10}
    do
        begin=$(date +%s)
        echo 'tasks : '$tasks, 'maps: '$maps
        bin/hadoop jar /home/hadoop/hadoop-3.3.3/share/hadoop/tools/lib/hadoop-streaming-3.3.3.jar -D mapred.reduce.tasks=$tasks -D mapreduce.job.maps=$maps -file ~/scripts/amapper3.3.py    -mapper ~/scripts/amapper3.3.py -file ~/scripts/areducer3.3.py   -reducer ~/scripts/areducer3.3.py -input /temp/10m_merged.csv  -output /myoutputs/3.3_tm_$tasks$maps &>> output_test1.txt
        #printf '\n' $>> output_test1.txt
        end=$(date +%s)
        tottime=$(expr $end - $begin)
        echo 'total time is: '$tottime
        echo 'end time is: '$end
    done
done

