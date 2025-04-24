import React from 'react';
import { Box, Stat, StatLabel, StatNumber, SimpleGrid, Heading } from '@chakra-ui/react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// Espera-se que metricsHistory seja um objeto do tipo:
// {
//   box_loss: [{epoch: 1, value: 0.8}, ...],
//   cls_loss: [...],
//   ...
// }
// currentMetrics é um objeto com os valores atuais das métricas

const COLORS = {
  box_loss: '#3182ce',
  cls_loss: '#38a169',
  dfl_loss: '#d69e2e',
  GPU_mem: '#805ad5',
  mAP50: '#e53e3e',
  mAP50_95: '#319795',
};

const METRIC_LABELS = {
  box_loss: 'Box Loss',
  cls_loss: 'Class Loss',
  dfl_loss: 'DFL Loss',
  GPU_mem: 'GPU Mem (GB)',
  mAP50: 'mAP50',
  mAP50_95: 'mAP50-95',
};

const MetricsPanel = ({ metricsHistory = {}, currentMetrics = {} }) => {
  // Exibir apenas as métricas que realmente existem no histórico ou nos valores atuais
  const allMetricKeys = Object.keys(METRIC_LABELS);
  const presentMetricKeys = allMetricKeys.filter(
    key => (metricsHistory[key] && metricsHistory[key].length > 0) || currentMetrics[key] !== undefined
  );

  return (
    <Box mb={8}>
      <Heading size="md" mb={4}>Metrics in Real Time</Heading>
      <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
        {presentMetricKeys.map((key) => (
          <Stat key={key} p={4} shadow="sm" border="1px" borderColor="gray.200" borderRadius="md" bg="white">
            <StatLabel>{METRIC_LABELS[key]}</StatLabel>
            <StatNumber>
              {currentMetrics[key] !== undefined && currentMetrics[key] !== null
                ? Number(currentMetrics[key]).toFixed(4)
                : '--'}
            </StatNumber>
            <Box height="60px" mt={2}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metricsHistory[key] || []}>
                  <XAxis dataKey="epoch" hide />
                  <YAxis domain={['auto', 'auto']} hide />
                  <Tooltip formatter={(v) => Number(v).toFixed(4)} labelFormatter={(e) => `Epoch: ${e}`} />
                  <Line type="monotone" dataKey="value" stroke={COLORS[key] || '#666'} dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Stat>
        ))}
      </SimpleGrid>
    </Box>
  );
};

export default MetricsPanel;
