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
            options {
                timeout(time: 30, unit: 'MINUTES')
            }
            steps {
                script {
                    retry(3) {
                        sh """
                            cd /mnt/mantle_data/jenkins/nzvm/nzcvm_data

                            # Fix the dubious ownership issue
                            git config --local --add safe.directory /mnt/mantle_data/jenkins/nzvm/nzcvm_data

                            git config pull.rebase false

                            # To test the latest main branch of nzcvm_data
                            git pull origin main

                            # To test a specific commit, comment out above, and use below

#                            # 1. Fetch latest history
#                            git fetch --all
#
#                            # 2. TIME TRAVEL: Force checkout a specific commit
#                            git checkout <commit hash>

                            # Clear any partial LFS downloads
                            git lfs prune

                            # Pull with explicit timeout
                            timeout 20m git lfs pull

                            # Verify critical files exist and aren't LFS pointers
                            echo "Verifying LFS files..."
                            if find . -name "*.h5" -exec file {} \\; | grep -q "ASCII text"; then
                                echo "ERROR: Found LFS pointer files instead of actual data"
                                exit 1
                            fi

                            # Ensure we have actual data files
                            data_size=\$(du -sm . | cut -f1)
                            echo "Data directory size: \${data_size}MB"
                            if [ "\$data_size" -lt 100 ]; then  # Adjust threshold as needed
                                echo "ERROR: Data directory too small (\${data_size}MB), likely incomplete"
                                exit 1
                            fi
                        """
                    }
                }
            }
        }

        stage('Run Tests in Container') {
            agent {
                docker { image 'earthquakesuc/nzvm'
                    // The -u 0 flags means run commands inside the container
                    // as the user with uid = 0. This user is, by default, the
                    // root user. So it is effectively saying run the commands
                    // as root.
                    args "-u 0 -v /mnt/mantle_data/jenkins/nzvm/Data:/nzvm/Data -v /mnt/mantle_data/jenkins/nzvm/benchmarks:/nzvm/benchmarks -v /mnt/mantle_data/jenkins/nzvm/nzcvm_data:/nzvm/nzcvm_data"
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
                        script {
                            def test_output_dir = "${env.WORKSPACE}/test_output-${env.BUILD_ID}"
                            withEnv(["JENKINS_OUTPUT_DIR=${test_output_dir}"]) {
                                sh """
                                    cd ${env.WORKSPACE}
                                    source .venv/bin/activate
                                    # Verify mount exists before creating symlink
                                    if [ ! -d "/nzvm/nzcvm_data" ]; then
                                        echo "ERROR: Mount /nzvm/nzcvm_data does not exist"
                                        exit 1
                                    fi
                                    # Check mount is not empty
                                    if [ ! "\$(ls -A /nzvm/nzcvm_data 2>/dev/null)" ]; then
                                        echo "ERROR: Mount /nzvm/nzcvm_data is empty"
                                        ls -la /nzvm/ || echo "Could not list /nzvm/"
                                        exit 1
                                    fi

                                    rm -f ${env.WORKSPACE}/velocity_modelling/nzcvm_data
                                    ln -s /nzvm/nzcvm_data ${env.WORKSPACE}/velocity_modelling/nzcvm_data

                                    # Verify symlink works
                                    if [ ! -r "${env.WORKSPACE}/velocity_modelling/nzcvm_data" ]; then
                                        echo "ERROR: Symlink is not readable"
                                        ls -la ${env.WORKSPACE}/velocity_modelling/
                                        exit 1
                                    fi

                                    # Remove all .pyc files and __pycache__ directories
                                    echo "Cleaning up .pyc files and __pycache__ directories..."
                                    find . -type f -name '*.pyc' -delete
                                    find . -type d -name '__pycache__' -delete

                                    # Run tests with PYTHONDONTWRITEBYTECODE
                                    export PYTHONDONTWRITEBYTECODE=1

                                    echo "Data verification passed, starting tests..."
                                    # Create the unique test output directory
                                    mkdir -p ${test_output_dir}
                                    pytest -s tests/ --benchmark-dir /nzvm/benchmarks --nzvm-binary-path /nzvm/NZVM --data-root ${env.WORKSPACE}/velocity_modelling/nzcvm_data
                                """
                            }
                        }
                    }
                    post {
                        failure {
                            script {
                                def test_output_dir = "${env.WORKSPACE}/test_output-${env.BUILD_ID}"
                                archiveArtifacts artifacts: "test_output-${env.BUILD_ID}/**", allowEmptyArchive: true
                            }
                        }
                        success {
                            script {
                                // Only cleanup on success
                                def test_output_dir = "${env.WORKSPACE}/test_output-${env.BUILD_ID}"
                                sh "rm -rf ${test_output_dir}"
                            }
                        }
                    }
                } // stage('Run regression tests')
            } // stages
        } // stage('Run Tests in Container')
    } // stages
} // pipeline
