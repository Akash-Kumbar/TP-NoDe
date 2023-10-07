echo preparing output for paper
echo On PUGAN dataset

echo Adding Gaussian BQ 0.5 256
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 256 --radius 0.5 --seed_k 3 --noise_type Gaussian --save_path data/Final/BQ/Gaussian/PS256_0_5/
echo Adding Gaussian BQ 1.0 512
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 512 --radius 1.0 --seed_k 3 --noise_type Gaussian --save_path data/Final/BQ/Gaussian/PS512_1_0/


echo Adding Laplacian BQ 0.5 256
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 256 --radius 0.5 --seed_k 3 --noise_type Laplacian --save_path data/Final/BQ/Laplacian/PS256_0_5/
echo Adding Laplacian BQ 1.0 512
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 512 --radius 1.0 --seed_k 3 --noise_type Laplacian --save_path data/Final/BQ/Laplacian/PS512_1_0/


echo Adding Discrete BQ 0.5 256
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 256 --radius 0.5 --seed_k 3 --noise_type Discrete --save_path data/Final/BQ/Discrete/PS256_0_5/
echo Adding Laplacian BQ 1.0 512
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 512 --radius 1.0 --seed_k 3 --noise_type Discrete --save_path data/Final/BQ/Discrete/PS512_1_0/

echo Adding UniformBall BQ 0.5 256
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 256 --radius 0.5 --seed_k 3 --noise_type UniformBall --save_path data/Final/BQ/UniformBall/PS256_0_5/
echo Adding UniformBall BQ 1.0 512
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 512 --radius 1.0 --seed_k 3 --noise_type UniformBall --save_path data/Final/BQ/UniformBall/PS512_1_0/

echo Adding Covariance BQ 0.5 256
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 256 --radius 0.5 --seed_k 3 --noise_type Covariance --save_path data/Final/BQ/Covariance/PS256_0_5/
echo Adding Covariance BQ 1.0 512
python upSampleWithNoise.py --noising BQ --upsampling_factor 4 --patch_size 512 --radius 1.0 --seed_k 3 --noise_type Covariance --save_path data/Final/BQ/Covariance/PS512_1_0/

echo adding UniformBall KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 64 --seed_k 6 --noise_type UniformBall --save_path data/Final/KNN/UniformBall/PS64/


echo Adding Gaussian DBQ 0.03
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.03 --seed_k 5 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS003/

echo Adding Gaussian DBQ 0.05
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.05 --seed_k 5 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS005/

echo Adding Gaussian DBQ 0.07
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.07 --seed_k 5 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS007/

echo Adding Gaussian DBQ 0.1
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.1 --seed_k 3 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS01/

echo Adding Gaussian DBQ 0.3
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.3 --seed_k 3 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS03/

echo Adding Gaussian DBQ 0.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.5 --seed_k 3 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS05/

echo Adding Gaussian DBQ 1.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.0 --seed_k 3 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS10/

echo Adding Gaussian DBQ 1.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.5 --seed_k 3 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS15/

echo Adding Gaussian DBQ 2.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 2.0 --seed_k 3 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS20/

echo adding Laplacian Global
python upSampleWithNoise.py --noising global --upsampling_factor 4 --patch_size 64 --seed_k 6 --noise_type Laplacian --save_path data/Final/Global/Laplacian/PS64/
echo Adding Laplacian Global
python upSampleWithNoise.py --noising global --upsampling_factor 4 --patch_size 128 --seed_k 4 --noise_type Laplacian --save_path data/Final/Global/Laplacian/PS128/
echo Adding Laplacian Global
python upSampleWithNoise.py --noising global --upsampling_factor 4 --patch_size 256 --seed_k 3 --noise_type Laplacian --save_path data/Final/Global/Laplacian/PS256/
echo Adding Laplacian Global
python upSampleWithNoise.py --noising global --upsampling_factor 4 --patch_size 512 --seed_k 3 --noise_type Laplacian --save_path data/Final/Global/Laplacian/PS512/


echo adding Laplacian KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 64 --seed_k 6 --noise_type Laplacian --save_path data/Final/KNN/Laplacian/PS64/
echo Adding Laplacian KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 128 --seed_k 4 --noise_type Laplacian --save_path data/Final/KNN/Laplacian/PS128/
echo Adding Laplacian KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 256 --seed_k 3 --noise_type Laplacian --save_path data/Final/KNN/Laplacian/PS256/
echo Adding Laplacian KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 512 --seed_k 3 --noise_type Laplacian --save_path data/Final/KNN/Laplacian/PS512/




echo Adding Gaussian DBQ 1.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.0 --seed_k 3 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS10/
echo Adding Gaussian DBQ 1.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.5 --seed_k 3 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS15/
echo Adding Gaussian DBQ 2.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 2.0 --seed_k 3 --noise_type Gaussian --save_path data/Final/Dilated_Bq/Gaussian/PS20/


echo Adding Laplacian DBQ 0.03
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.03 --seed_k 5 --noise_type Laplacian --save_path data/Final/Dilated_Bq/Laplacian/PS003/
echo Adding Laplacian DBQ 0.05
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.05 --seed_k 5 --noise_type Laplacian --save_path data/Final/Dilated_Bq/Laplacian/PS005/
echo Adding Laplacian DBQ 0.07
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.07 --seed_k 5 --noise_type Laplacian --save_path data/Final/Dilated_Bq/Laplacian/PS007/
echo Adding Laplacian DBQ 0.1
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.1 --seed_k 3 --noise_type Laplacian --save_path data/Final/Dilated_Bq/Laplacian/PS01/
echo Adding Laplacian DBQ 0.3
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.3 --seed_k 3 --noise_type Laplacian --save_path data/Final/Dilated_Bq/Laplacian/PS03/
echo Adding Laplacian DBQ 0.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.5 --seed_k 3 --noise_type Laplacian --save_path data/Final/Dilated_Bq/Laplacian/PS05/
echo Adding Laplacian DBQ 1.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.0 --seed_k 3 --noise_type Laplacian --save_path data/Final/Dilated_Bq/Laplacian/PS10/
echo Adding Laplacian DBQ 1.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.5 --seed_k 3 --noise_type Laplacian --save_path data/Final/Dilated_Bq/Laplacian/PS15/
echo Adding Laplacian DBQ 2.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 2.0 --seed_k 3 --noise_type Laplacian --save_path data/Final/Dilated_Bq/Laplacian/PS20/





echo Adding Discrete DBQ 0.03
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.03 --seed_k 5 --noise_type Discrete --save_path data/Final/Dilated_Bq/Discrete/PS003/
echo Adding Discrete DBQ 0.05
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.05 --seed_k 5 --noise_type Discrete --save_path data/Final/Dilated_Bq/Discrete/PS005/
echo Adding Discrete DBQ 0.07
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.07 --seed_k 5 --noise_type Discrete --save_path data/Final/Dilated_Bq/Discrete/PS007/
echo Adding Discrete DBQ 0.1
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.1 --seed_k 3 --noise_type Discrete --save_path data/Final/Dilated_Bq/Discrete/PS01/
echo Adding Discrete DBQ 0.3
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.3 --seed_k 3 --noise_type Discrete --save_path data/Final/Dilated_Bq/Discrete/PS03/
echo Adding Discrete DBQ 0.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.5 --seed_k 3 --noise_type Discrete --save_path data/Final/Dilated_Bq/Discrete/PS05/
echo Adding Discrete DBQ 1.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.0 --seed_k 3 --noise_type Discrete --save_path data/Final/Dilated_Bq/Discrete/PS10/
echo Adding Discrete DBQ 1.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.5 --seed_k 3 --noise_type Discrete --save_path data/Final/Dilated_Bq/Discrete/PS15/
echo Adding Discrete DBQ 2.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 2.0 --seed_k 3 --noise_type Discrete --save_path data/Final/Dilated_Bq/Discrete/PS20/





echo Adding UniformBall DBQ 0.03
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.03 --seed_k 5 --noise_type UniformBall --save_path data/Final/Dilated_Bq/UniformBall/PS003/
echo Adding UniformBall DBQ 0.05
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.05 --seed_k 5 --noise_type UniformBall --save_path data/Final/Dilated_Bq/UniformBall/PS005/
echo Adding UniformBall DBQ 0.07
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.07 --seed_k 5 --noise_type UniformBall --save_path data/Final/Dilated_Bq/UniformBall/PS007/
echo Adding UniformBall DBQ 0.1
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.1 --seed_k 3 --noise_type UniformBall --save_path data/Final/Dilated_Bq/UniformBall/PS01/
echo Adding UniformBall DBQ 0.3
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.3 --seed_k 3 --noise_type UniformBall --save_path data/Final/Dilated_Bq/UniformBall/PS03/
echo Adding UniformBall DBQ 0.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.5 --seed_k 3 --noise_type UniformBall --save_path data/Final/Dilated_Bq/UniformBall/PS05/
echo Adding UniformBall DBQ 1.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.0 --seed_k 3 --noise_type UniformBall --save_path data/Final/Dilated_Bq/UniformBall/PS10/
echo Adding UniformBall DBQ 1.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.5 --seed_k 3 --noise_type UniformBall --save_path data/Final/Dilated_Bq/UniformBall/PS15/
echo Adding UniformBall DBQ 2.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 2.0 --seed_k 3 --noise_type UniformBall --save_path data/Final/Dilated_Bq/UniformBall/PS20/




echo Adding Covariance DBQ 0.03
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.03 --seed_k 5 --noise_type Covariance --save_path data/Final/Dilated_Bq/Covariance/PS003/
echo Adding Covariance DBQ 0.05
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.05 --seed_k 5 --noise_type Covariance --save_path data/Final/Dilated_Bq/Covariance/PS005/
echo Adding Covariance DBQ 0.07
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.07 --seed_k 5 --noise_type Covariance --save_path data/Final/Dilated_Bq/Covariance/PS007/
echo Adding Covariance DBQ 0.1
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.1 --seed_k 3 --noise_type Covariance --save_path data/Final/Dilated_Bq/Covariance/PS01/
echo Adding Covariance DBQ 0.3
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.3 --seed_k 3 --noise_type Covariance --save_path data/Final/Dilated_Bq/Covariance/PS03/
echo Adding Covariance DBQ 0.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 0.5 --seed_k 3 --noise_type Covariance --save_path data/Final/Dilated_Bq/Covariance/PS05/
echo Adding Covariance DBQ 1.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.0 --seed_k 3 --noise_type Covariance --save_path data/Final/Dilated_Bq/Covariance/PS10/
echo Adding Covariance DBQ 1.5
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 1.5 --seed_k 3 --noise_type Covariance --save_path data/Final/Dilated_Bq/Covariance/PS15/
echo Adding Covariance DBQ 2.0
python upSampleWithNoise.py --noising dilated_BQ --upsampling_factor 4 --radius 2.0 --seed_k 3 --noise_type Covariance --save_path data/Final/Dilated_Bq/Covariance/PS20/





