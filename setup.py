import subprocess
import sys


def install(*packages):
    command = [sys.executable, '-m', 'pip', 'install']
    command.extend(packages)
    subprocess.run(command)


if __name__ == '__main__':
    install('torch', 'torchvision', 'torchaudio', '--extra-index-url', 'https://download.pytorch.org/whl/cu113')
    install('monai[all]', '-U')
    install('monai', '-U')
    install('numpy')
    install('pandas')
    install('pytorch-ignite')
