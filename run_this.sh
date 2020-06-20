sudo apt update
sudo apt install ffmpeg
ffmpeg -version

sudo apt-get install python-pip
sudo pip install virtualenv
mkdir ~/.virtualenvs
sudo pip install virtualenvwrapper
export WORKON_HOME=~/.virtualenvs
echo -e "./usr/local/bin/virtualenvwrapper.sh"  >> ~/.bashrc
mkvirtualenv speed

