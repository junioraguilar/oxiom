import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Card,
  CardHeader,
  CardBody,
  HStack,
  Wrap,
  WrapItem,
  Tag,
  Heading,
  Text,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  Button,
  Spinner,
  Alert,
  AlertIcon,
} from '@chakra-ui/react';
import { DownloadIcon, DeleteIcon } from '@chakra-ui/icons';
import DatasetUploadModal from '../components/DatasetUploadModal';

interface Dataset {
  id: string;
  name: string;
  created_at: string;
  file_size: number;
  classes: string[];
}

const DatasetsPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string[]>([]);
  const [isDatasetModalOpen, setIsDatasetModalOpen] = useState(false);

  const handleOpenDatasetModal = () => setIsDatasetModalOpen(true);
  const handleCloseDatasetModal = () => setIsDatasetModalOpen(false);
  const handleUploadComplete = () => {
    setIsDatasetModalOpen(false);
    fetchDatasets();
  };

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const res = await axios.get('http://localhost:5000/api/datasets');
      setDatasets(res.data.datasets || []);
      setError(null);
    } catch (err: any) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleDownload = (id: string) => {
    window.open(`http://localhost:5000/api/datasets/${id}/download`, '_blank');
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;
    setDeleting(prev => [...prev, id]);
    try {
      await axios.delete(`http://localhost:5000/api/datasets/${id}/delete`);
      fetchDatasets();
    } catch (err) {
      console.error('Error deleting dataset:', err);
    } finally {
      setDeleting(prev => prev.filter(x => x !== id));
    }
  };

  return (
    <>
      <Card variant="outline" width="100%" mb={4}>
        <CardHeader pb={2}>
          <HStack justify="space-between">
            <Heading size="md">Datasets</Heading>
            <HStack spacing={2}>
              <Button size="sm" onClick={fetchDatasets} isLoading={loading}>
                Refresh
              </Button>
              <Button size="sm" onClick={handleOpenDatasetModal}>
                Upload Dataset
              </Button>
            </HStack>
          </HStack>
        </CardHeader>
        <CardBody>
          {error && (
            <Alert status="error" mb={4}>
              <AlertIcon /> {error}
            </Alert>
          )}
          {loading ? (
            <Spinner />
          ) : datasets.length === 0 ? (
            <Text>No datasets found.</Text>
          ) : (
            <TableContainer>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th>Name</Th>
                    <Th>Created At</Th>
                    <Th>Size</Th>
                    <Th>Classes</Th>
                    <Th>Actions</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {datasets.map(ds => (
                    <Tr key={ds.id}>
                      <Td>{ds.name}</Td>
                      <Td>{new Date(ds.created_at).toLocaleString()}</Td>
                      <Td>{(ds.file_size / (1024 * 1024)).toFixed(2)} MB</Td>
                      <Td>
                        <Wrap spacing={1}>
                          {ds.classes.map(cls => (
                            <WrapItem key={cls}>
                              <Tag size="sm" variant="subtle" colorScheme="blue">{cls}</Tag>
                            </WrapItem>
                          ))}
                        </Wrap>
                      </Td>
                      <Td>
                        <HStack spacing={2}>
                          <Button
                            size="xs"
                            leftIcon={<DownloadIcon />}
                            colorScheme="blue"
                            onClick={() => handleDownload(ds.id)}
                          >
                            Download
                          </Button>
                          <Button
                            size="xs"
                            leftIcon={<DeleteIcon />}
                            colorScheme="red"
                            isLoading={deleting.includes(ds.id)}
                            loadingText="Deleting"
                            onClick={() => handleDelete(ds.id)}
                          >
                            Delete
                          </Button>
                        </HStack>
                      </Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </TableContainer>
          )}
        </CardBody>
      </Card>
      <DatasetUploadModal
        isOpen={isDatasetModalOpen}
        onClose={handleCloseDatasetModal}
        onUploadComplete={handleUploadComplete}
      />
    </>
  );
};

export default DatasetsPage;
