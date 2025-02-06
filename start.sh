sudo touch 'echomimic.log'
sudo docker build -t echomimic_v2:latest .
sudo docker kill echomimic_v2
sudo docker rm echomimic_v2
sudo docker run --it --gpus '"device=1"' --name=echomimic_v2 -v "/opt/hzai-echomimic_v2/mount/Youtube_crawled" --network host echomimic_v2:latest