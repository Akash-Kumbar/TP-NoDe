## echo preparing output for paper
echo On PUGAN dataset

echo Adding Gaussian KNN 64
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 64 --seed_k 6 --noise_type Gaussian --save_path data/Final/KNN/Gaussian/PS64/

echo Adding Gaussian KNN 128
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 128 --seed_k 4 --noise_type Gaussian --save_path data/Final/KNN/Gaussian/PS128/

echo Adding Gaussian KNN 256
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 256 --seed_k 3 --noise_type Gaussian --save_path data/Final/KNN/Gaussian/PS256/

echo Adding Gaussian KNN 512
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 512 --seed_k 3 --noise_type Gaussian --save_path data/Final/KNN/Gaussian/PS512/



echo adding Discrete KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 64 --seed_k 6 --noise_type Discrete --save_path data/Final/KNN/Discrete/PS64/

echo Adding Discrete KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 128 --seed_k 4 --noise_type Discrete --save_path data/Final/KNN/Discrete/PS128/

echo Adding Discrete KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 256 --seed_k 3 --noise_type Discrete --save_path data/Final/KNN/Discrete/PS256/

echo Adding Discrete KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 512 --seed_k 3 --noise_type Discrete --save_path data/Final/KNN/Discrete/PS512/



echo adding UniformBall KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 64 --seed_k 6 --noise_type UniformBall --save_path data/Final/KNN/UniformBall/PS64/

echo Adding UniformBall KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 128 --seed_k 4 --noise_type UniformBall --save_path data/Final/KNN/UniformBall/PS128/

echo Adding UniformBall KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 256 --seed_k 3 --noise_type UniformBall --save_path data/Final/KNN/UniformBall/PS256/

echo Adding UniformBall KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 512 --seed_k 3 --noise_type UniformBall --save_path data/Final/KNN/UniformBall/PS512/



echo adding Covariance KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 64 --seed_k 6 --noise_type Covariance --save_path data/Final/KNN/Covariance/PS64/

echo Adding Covariance KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 128 --seed_k 4 --noise_type Covariance --save_path data/Final/KNN/Covariance/PS128/

echo Adding Covariance KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 256 --seed_k 3 --noise_type Covariance --save_path data/Final/KNN/Covariance/PS256/

echo Adding Covariance KNN
python upSampleWithNoise.py --noising KNN --upsampling_factor 4 --patch_size 512 --seed_k 3 --noise_type Covariance --save_path data/Final/KNN/Covariance/PS512/



