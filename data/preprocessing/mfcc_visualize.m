[y, fs] = audioread('C:\Users\Franz Kaiser\Nextcloud\0Studium\Dateien\9. Semester\Speech Recognition\Project\source\data\preprocessing\SX83.wav');
y = y(1:32000);
mfccs = mfcc(y, fs, 'WindowLength', 320, 'OverlapLength', 160, 'NumCoeffs', 29);
imagesc(mfccs')

