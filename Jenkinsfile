pipeline {
    agent none 

    stages {
        stage('Pull Latest Docker Image') {
            agent any
            steps {
                sh 'docker pull earthquakesuc/nzvm'
            }
        }

        stage('Update nzcvm_data Repository') {
            agent any
            steps {
                sh """
                    cd /mnt/mantle_data/jenkins/nzvm/nzcvm_data
                    git pull origin main
                    git lfs pull
                """
            }
        }

        stage('Run Tests in Container') {
            agent {
                docker { image 'earthquakesuc/nzvm'
                    // The -u 0 flags means run commands inside the container
                    // as the user with uid = 0. This user is, by default, the
                    // root user. So it is effectively saying run the commands
                    // as root.
                    args "-u 0 -v /mnt/mantle_data/jenkins/nzvm/Data:/nzvm/Data -v /mnt/mantle_data/jenkins/nzvm/benchmarks:/nzvm/benchmarks -v /mnt/mantle_data/jenkins/nzvm/nzcvm_data:/nzvm/nzcvm_data -v /tmp/jenkins:/tmp"
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
                            rm -f ${env.WORKSPACE}/velocity_modelling/nzcvm_data
                            ln -s /nzvm/nzcvm_data ${env.WORKSPACE}/velocity_modelling/nzcvm_data
                            pytest -s tests/ --benchmark-dir /nzvm/benchmarks --nzvm-binary-path /nzvm/NZVM --data-root ${env.WORKSPACE}/velocity_modelling/nzcvm_data
                        """
                    }
                }
            }
        } // stage('Run Tests in Container')
    } // stages
} // pipeline
