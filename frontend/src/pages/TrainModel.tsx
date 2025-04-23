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
  useToast,
  Select,
  Flex,
  HStack
} from '@chakra-ui/react'
import api from '../api/axios'
import TrainingProgress from '../components/TrainingProgress'

// Interface for dataset object
interface Dataset {
  id: string;
  name: string;
  file_path: string;
  yaml_path: string;
  created_at: string;
  file_size: number;
  classes: string[];
}

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
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDatasetId, setSelectedDatasetId] = useState('')
  const [loadingDatasets, setLoadingDatasets] = useState(false)
  
  const toast = useToast()

  // Add useEffect to check for ongoing training when component mounts
  useEffect(() => {
    const checkOngoingTraining = async () => {
      try {
        const response = await api.get('/api/training-status')
        
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
    
    fetchDatasets()
    checkOngoingTraining()
  }, [toast])

  // Fetch datasets from the API
  const fetchDatasets = async () => {
    setLoadingDatasets(true)
    try {
      const response = await api.get('/api/datasets')
      setDatasets(response.data.datasets)
    } catch (error) {
      console.error('Error fetching datasets:', error)
      toast({
        title: 'Erro',
        description: 'Não foi possível carregar os datasets',
        status: 'error',
        duration: 3000,
        isClosable: true,
      })
    } finally {
      setLoadingDatasets(false)
    }
  }

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
      setSelectedDatasetId('');
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
      const response = await api.post('/api/upload-dataset', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 100)
          )
          setUploadProgress(percentCompleted)
        },
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
      
      // Refresh datasets list after upload
      fetchDatasets()
    } catch (error: any) {
      console.error('Upload falhou:', error)
      
      let errorMsg = 'Falha ao enviar dataset'
        if (error.code === 'ECONNABORTED') {
        errorMsg = 'O upload expirou. O arquivo pode ser muito grande.'
        } else if (error.response) {
        errorMsg = error.response.data?.error || errorMsg
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

  const handleDatasetSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value
    setSelectedDatasetId(id)
    if (id) {
      setDatasetId(id)
      setUploadSuccess(true)
      setError('')
    } else {
      setDatasetId('')
      setUploadSuccess(false)
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

    try {
      const response = await api.post('/api/train', {
        dataset_id: datasetId,
        epochs: epochs,
        batch_size: batchSize,
        img_size: imageSize
      })

      setTrainingSuccess(true)
      setModelId(response.data.model_id)
    } catch (error: any) {
      console.error('Training failed:', error)
      
      let errorMsg = 'Error starting training'
      if (error.response) {
        errorMsg = error.response.data?.error || errorMsg
      }
      
      setError(errorMsg)
      setIsTraining(false)
      
      toast({
        title: 'Training failed',
        description: errorMsg,
        status: 'error',
        duration: 5000,
        isClosable: true,
      })
    }
  }

  if (trainingSuccess && modelId) {
    return (
      <Box p={6}>
        <Heading size="lg" mb={6}>Training Progress</Heading>
        <TrainingProgress modelId={modelId} onCompleted={() => setIsTraining(false)} />
      </Box>
    )
  }

  return (
    <Box p={6}>
      <Heading size="lg" mb={6}>Train YOLOv8 Model</Heading>
      
      <VStack spacing={6} align="stretch">
        <Box p={5} borderWidth="1px" borderRadius="lg">
          <Heading size="md" mb={4}>Select Dataset</Heading>
        
          <FormControl mb={4}>
            <FormLabel>Available Datasets</FormLabel>
            <Select 
              placeholder="Select a dataset" 
              value={selectedDatasetId}
              onChange={handleDatasetSelect}
              isDisabled={loadingDatasets}
            >
              {datasets.map(dataset => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.classes.join(', ')})
                </option>
              ))}
            </Select>
            <FormHelperText>
              Select a dataset you previously uploaded or upload a new one
            </FormHelperText>
          </FormControl>
          
          <Text fontWeight="medium" mb={2}>Or upload a new dataset:</Text>
            
            <FormControl>
            <FormLabel>Dataset (ZIP file)</FormLabel>
              <Input
                type="file"
                accept=".zip"
                onChange={handleFileChange}
              disabled={isUploading}
              p={1}
              />
              <FormHelperText>
              Upload a ZIP file containing your YOLO dataset
              </FormHelperText>
            </FormControl>
            
            {isUploading && (
            <Box mt={4}>
              <Progress value={uploadProgress} size="sm" colorScheme="blue" />
              <FormControl>
                <FormHelperText textAlign="center">{uploadProgress}% completed</FormHelperText>
              </FormControl>
              </Box>
            )}
            
          <Flex justifyContent="flex-end" mt={4}>
            <Button 
              colorScheme="blue" 
              onClick={handleUploadDataset} 
              isLoading={isUploading} 
              loadingText="Uploading..."
              disabled={!selectedFile || isUploading || !!selectedDatasetId}
            >
              Upload Dataset
            </Button>
          </Flex>
        </Box>
            
        {uploadSuccess && (
          <Box p={5} borderWidth="1px" borderRadius="lg">
            <Heading size="md" mb={4}>Training Parameters</Heading>
            
            <HStack spacing={4} align="flex-start" mb={4}>
            <FormControl>
                <FormLabel>Epochs</FormLabel>
              <NumberInput 
                value={epochs} 
                onChange={(_, value) => setEpochs(value)}
                  min={1} 
                  max={500}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <FormHelperText>
                  Number of training epochs
              </FormHelperText>
            </FormControl>
            
            <FormControl>
              <FormLabel>Batch Size</FormLabel>
              <NumberInput 
                value={batchSize} 
                onChange={(_, value) => setBatchSize(value)}
                  min={1} 
                  max={64}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <FormHelperText>
                  Training batch size
              </FormHelperText>
            </FormControl>
            
            <FormControl>
              <FormLabel>Image Size</FormLabel>
              <NumberInput 
                  value={imageSize} 
                  onChange={(_, value) => setImageSize(value)} 
                min={320} 
                max={1280} 
                step={32}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <FormHelperText>
                  Input image size (px)
              </FormHelperText>
            </FormControl>
            </HStack>
            
            <Flex justifyContent="flex-end">
            <Button 
                colorScheme="blue" 
              onClick={handleTrainModel} 
              isLoading={isTraining} 
                loadingText="Starting training..."
                disabled={isTraining}
            >
              Start Training
            </Button>
            </Flex>
        </Box>
        )}
        
        {error && (
          <Alert status="error">
            <AlertIcon />
            <AlertTitle mr={2}>Error!</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </VStack>
    </Box>
  )
}

export default TrainModel 