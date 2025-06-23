pipeline {
    agent none

    stages {
        stage('Pull Latest Docker Image') {
            agent any
            steps {
                sh 'docker pull earthquakesuc/nzvm'
            }
        }

        stage('Run in Docker') {
            agent {
                docker {
                    image 'earthquakesuc/nzvm'
                    args "-u 0 -v /mnt/mantle_data/jenkins/nzvm/Data:/nzvm/Data -v /mnt/mantle_data/jenkins/nzvm/benchmarks:/nzvm/benchmarks -v /mnt/mantle_data/jenkins/nzvm/global:/nzvm/global"
                }
            }
            stages {
                stage('Install Python') {
                    steps {
                        sh """
                            apt-get update
                            apt-get install -y python3-pip python3-venv python3
                        """
                    }
                }
                stage('Installing OS Dependencies') {
                    steps {
                        echo "[[ Install GMT ]]"
                        sh """
                            apt-get update
                            apt-get install -y gmt libgmt-dev libgmt6 ghostscript
                        """
                    }
                }
                stage('Install UV') {
                    steps {
                        sh """
                            curl -LsSf https://astral.sh/uv/install.sh | sh
                        """
                    }
                }
                stage('Setting up env') {
                    steps {
                        echo "[[ Start virtual environment ]]"
                        sh """
                            source ~/.local/bin/env sh
                            cd ${env.WORKSPACE}
                            uv venv
                            source .venv/bin/activate
                            uv pip install -e .
                        """
                    }
                }
                stage('Run regression tests') {
                    steps {
                        sh """
                            cd ${env.WORKSPACE}
                            source .venv/bin/activate
                            cp -r /nzvm/global/* ${env.WORKSPACE}/velocity_modelling/data/global
                            pytest -s tests/ --benchmark-dir /nzvm/benchmarks --nzvm-binary-path /nzvm/NZVM
                        """
                    }
                }
            }
        }
    }
}

