import { useState, useEffect } from 'react'
import {
  Box,
  Heading,
  Text,
  VStack,
  Button,
  FormControl,
  FormLabel,
  FormHelperText,
  Input,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Progress,
  Divider,
  useToast
} from '@chakra-ui/react'
import axios from 'axios'
import TrainingProgress from '../components/TrainingProgress'

const TrainModel = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [datasetId, setDatasetId] = useState('')
  const [trainingSuccess, setTrainingSuccess] = useState(false)
  const [modelId, setModelId] = useState('')
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(16)
  const [imageSize, setImageSize] = useState(640)
  const [error, setError] = useState('')
  
  const toast = useToast()

  // Add useEffect to check for ongoing training when component mounts
  useEffect(() => {
    const checkOngoingTraining = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/training-status')
        
        // Check if there are any active training sessions
        if (response.data.training_sessions && response.data.training_sessions.length > 0) {
          // Find the most recent non-completed, non-error training
          const activeTraining = response.data.training_sessions.find(
            (session: any) => 
              session.info.status !== 'completed' && 
              session.info.status !== 'error'
          )
          
          if (activeTraining) {
            // We found an ongoing training session
            setModelId(activeTraining.model_id)
            setTrainingSuccess(true)
            
            toast({
              title: 'Training in progress',
              description: `Reconnected to ongoing training for model: ${activeTraining.model_id}`,
              status: 'info',
              duration: 5000,
              isClosable: true,
            })
          }
        }
      } catch (error) {
        console.error('Error checking training status:', error)
      }
    }
    
    checkOngoingTraining()
  }, [toast])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      
      // Verificar se o arquivo é um ZIP
      if (!file.name.toLowerCase().endsWith('.zip')) {
        toast({
          title: 'Formato inválido',
          description: 'Por favor, selecione um arquivo ZIP',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
        return;
      }
      
      // Verificar o tamanho do arquivo (limite de 900MB para segurança)
      const maxSize = 900 * 1024 * 1024; // 900MB em bytes
      if (file.size > maxSize) {
        toast({
          title: 'Arquivo muito grande',
          description: 'O tamanho máximo do arquivo é 900MB',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
        return;
      }
      
      setSelectedFile(file);
      setUploadSuccess(false);
      setDatasetId('');
      setTrainingSuccess(false);
      setModelId('');
      setError('');
    }
  }

  const handleUploadDataset = async () => {
    if (!selectedFile) {
      toast({
        title: 'Nenhum arquivo selecionado',
        description: 'Por favor, selecione um arquivo de dataset para upload',
        status: 'error',
        duration: 3000,
        isClosable: true,
      })
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)

    setIsUploading(true)
    setUploadProgress(0)
    setError('')

    try {
      const response = await axios.post('/api/upload-dataset', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 100)
          )
          setUploadProgress(percentCompleted)
        },
        // Aumentando timeout para arquivos grandes
        timeout: 600000, // 10 minutos
      })

      setUploadSuccess(true)
      setDatasetId(response.data.dataset_id)
      
      toast({
        title: 'Upload bem-sucedido',
        description: 'Dataset foi enviado com sucesso',
        status: 'success',
        duration: 5000,
        isClosable: true,
      })
    } catch (error) {
      console.error('Upload falhou:', error)
      
      let errorMsg = 'Falha ao enviar dataset'
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED') {
          errorMsg = 'O upload expirou. O arquivo pode ser muito grande.';
        } else if (error.response) {
          if (error.response.status === 413) {
            errorMsg = 'Arquivo muito grande. O tamanho máximo permitido é 900MB.';
          } else {
            errorMsg = error.response.data.error || errorMsg;
          }
        }
      }
      
      setError(errorMsg)
      toast({
        title: 'Falha no upload',
        description: errorMsg,
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setIsUploading(false)
    }
  }

  const handleTrainModel = async () => {
    if (!datasetId) {
      toast({
        title: 'No dataset',
        description: 'Please upload a dataset first',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      })
      return
    }

    setIsTraining(true)
    setError('')
    setTrainingSuccess(false)
    setModelId('')

    try {
      const response = await axios.post('/api/train', {
        dataset_id: datasetId,
        epochs: epochs,
        batch_size: batchSize,
        img_size: imageSize
      })

      setTrainingSuccess(true)
      setModelId(response.data.model_id)
      
      toast({
        title: 'Training started',
        description: 'Model training has been initiated successfully',
        status: 'success',
        duration: 5000,
        isClosable: true,
      })
    } catch (error) {
      console.error('Training request failed:', error)
      
      let errorMsg = 'Failed to start model training'
      if (axios.isAxiosError(error) && error.response) {
        errorMsg = error.response.data.error || errorMsg
      }
      
      setError(errorMsg)
      toast({
        title: 'Training failed',
        description: errorMsg,
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <Box py={8}>
      <VStack spacing={8} align="stretch">
        <Heading as="h1" size="xl">
          Train Your YOLO Model
        </Heading>
        
        <Text>
          Upload your dataset and configure training parameters to train a custom YOLOv8 model.
        </Text>
        
        <Box p={6} borderWidth="1px" borderRadius="lg" boxShadow="sm">
          <VStack spacing={6} align="stretch">
            <Heading size="md">Step 1: Upload Dataset</Heading>
            
            <FormControl>
              <FormLabel htmlFor="dataset-file">Select Dataset File (ZIP)</FormLabel>
              <Input
                id="dataset-file"
                type="file"
                accept=".zip"
                onChange={handleFileChange}
                padding={1}
              />
              <FormHelperText>
                O dataset deve estar no formato YOLO, contendo:
                <ul style={{ marginLeft: '20px', marginTop: '5px' }}>
                  <li>Arquivo <strong>data.yaml</strong> com configuração de classes</li>
                  <li>Pasta <strong>images/</strong> com as imagens de treinamento</li>
                  <li>Pasta <strong>labels/</strong> com as anotações no formato YOLO</li>
                </ul>
              </FormHelperText>
            </FormControl>
            
            {selectedFile && (
              <Text fontSize="sm">
                Selected file: <strong>{selectedFile.name}</strong> ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
              </Text>
            )}
            
            {isUploading && (
              <Box>
                <Text mb={2}>Uploading... {uploadProgress}%</Text>
                <Progress value={uploadProgress} size="sm" colorScheme="blue" borderRadius="md" />
              </Box>
            )}
            
            {uploadSuccess && (
              <Alert status="success" borderRadius="md">
                <AlertIcon />
                <AlertTitle>Upload Successful!</AlertTitle>
                <AlertDescription>Dataset has been uploaded and is ready for training.</AlertDescription>
              </Alert>
            )}
            
            <Button 
              colorScheme="blue" 
              onClick={handleUploadDataset} 
              isLoading={isUploading} 
              loadingText="Uploading" 
              isDisabled={!selectedFile || isUploading}
            >
              Upload Dataset
            </Button>
            
            <Divider />
            
            <Heading size="md">Step 2: Configure Training</Heading>
            
            <FormControl>
              <FormLabel>Number of Epochs</FormLabel>
              <NumberInput 
                min={1} 
                max={300} 
                value={epochs} 
                onChange={(_, value) => setEpochs(value)}
                isDisabled={!uploadSuccess}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <FormHelperText>
                Higher values may improve accuracy but take longer to train
              </FormHelperText>
            </FormControl>
            
            <FormControl>
              <FormLabel>Batch Size</FormLabel>
              <NumberInput 
                min={1} 
                max={128} 
                value={batchSize} 
                onChange={(_, value) => setBatchSize(value)}
                isDisabled={!uploadSuccess}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <FormHelperText>
                Adjust based on your GPU memory
              </FormHelperText>
            </FormControl>
            
            <FormControl>
              <FormLabel>Image Size</FormLabel>
              <NumberInput 
                min={320} 
                max={1280} 
                step={32}
                value={imageSize} 
                onChange={(_, value) => setImageSize(value)}
                isDisabled={!uploadSuccess}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <FormHelperText>
                Larger sizes may improve detection of small objects
              </FormHelperText>
            </FormControl>
            
            {error && (
              <Alert status="error" borderRadius="md">
                <AlertIcon />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            {trainingSuccess && (
              <>
                <Alert status="success" borderRadius="md">
                  <AlertIcon />
                  <AlertTitle>Training Started!</AlertTitle>
                  <AlertDescription>
                    Your model training has been initiated. You can monitor progress below.
                  </AlertDescription>
                </Alert>
                
                <Divider my={4} />
                
                <Heading size="md">Training Progress</Heading>
                {modelId && (
                  <TrainingProgress modelId={modelId} />
                )}
              </>
            )}
            
            <Button 
              colorScheme="green" 
              onClick={handleTrainModel} 
              isLoading={isTraining} 
              loadingText="Starting Training" 
              isDisabled={!uploadSuccess || isTraining}
            >
              Start Training
            </Button>
          </VStack>
        </Box>
      </VStack>
    </Box>
  )
}

export default TrainModel 