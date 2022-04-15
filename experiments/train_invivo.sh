set -e

for dataset in A549 GM12878 HeLa-S3
do
    for regularizer in standard mixup manifold-mixup gn manifold-gn sn adversarial
    do
        for BN in --bn --nobn
        do
            for activation in relu exponential
            do
                qsub -N fixedlr_${regularizer}_${BN}_${activation} train_invivo_fixed.sh $dataset $regularizer $activation $BN
            done
        done
    done
done

for dataset in A549 GM12878 HeLa-S3
do
    for regularizer in standard mixup manifold-mixup
    do
        for BN in --bn
        do
            for activation in relu exponential
            do
                qsub -N warmup_${regularizer}_${BN}_${activation} train_invivo_warmup.sh $dataset $regularizer $activation $BN
            done
        done
    done
done