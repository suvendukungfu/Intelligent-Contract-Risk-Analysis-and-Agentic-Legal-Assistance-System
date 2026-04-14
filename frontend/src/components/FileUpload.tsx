import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, isLoading }) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
    },
    maxFiles: 1,
    disabled: isLoading,
  });

  return (
    <div
      {...getRootProps()}
      className={`file-upload ${isDragActive ? 'active' : ''}`}
    >
      <input {...getInputProps()} />
      <div className="file-upload-icon">📄</div>
      {isDragActive ? (
        <p className="file-upload-text">Drop the file here...</p>
      ) : (
        <>
          <p className="file-upload-text">
            Drag and drop a contract file here, or click to select
          </p>
          <p className="file-upload-hint">Supported formats: PDF, TXT (Max 10MB)</p>
        </>
      )}
    </div>
  );
};

export default FileUpload;
