mkdir ../data
cd ../data
mkdir dataset_2005_raw -p
cd dataset_2005_raw
mkdir Art -p
mkdir Books -p
mkdir Dolls -p
mkdir Laundry -p
mkdir Moebius -p
mkdir Reindeer -p

wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/Depth/art.bmp -O Art/disp1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/RGB/art.bmp -O Art/view1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/Depth/books.bmp -O Books/disp1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/RGB/books.bmp -O Books/view1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/Depth/dolls.bmp -O Dolls/disp1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/RGB/dolls.bmp -O Dolls/view1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/Depth/laundry.bmp -O Laundry/disp1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/RGB/laundry.bmp -O Laundry/view1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/Depth/moebius.bmp -O Moebius/disp1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/RGB/moebius.bmp -O Moebius/view1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/Depth/reindeer.bmp -O Reindeer/disp1.bmp
wget https://github.com/twhui/MSG-Net/raw/master/MSGNet-release/testing%20sets/B/RGB/reindeer.bmp -O Reindeer/view1.bmp
