pipeline {
    agent {
        docker { image 'earthquakesuc/nzvm'
            // The -u 0 flags means run commands inside the container
            // as the user with uid = 0. This user is, by default, the
            // root user. So it is effectively saying run the commands
            // as root.
            args "-u 0 -v /home/jenkins/Data:/nzvm/Data -v /home/jenkins/benchmarks:${env.WORKSPACE}/tests/benchmarks -v /home/jenkins/Data:${env.WORKSPACE}/velocity_modelling/Data"
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
                    pytest -s tests/
                """
            }
        }
    }
}