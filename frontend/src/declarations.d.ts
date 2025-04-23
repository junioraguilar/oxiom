declare module '*.jsx' {
  import React from 'react';
  const Component: React.FC<any>;
  export default Component;
}

declare module '*.png';
declare module '*.jpg';
declare module '*.jpeg';
declare module '*.svg';

declare module '../components/TrainingProgress';
declare module '../components/TrainingProgress.jsx';
declare module '../components/ModelsList';
declare module '../components/ModelsList.jsx';
declare module '../components/TrainingControls';
declare module '../components/TrainingControls.jsx'; 