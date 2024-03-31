for i in {1..7}
do
   ffmpeg -i "data/wildtrack/videos/cam${i}.mp4" -t 30 "data/wildtrack/videos30/cam${i}.mp4"
done