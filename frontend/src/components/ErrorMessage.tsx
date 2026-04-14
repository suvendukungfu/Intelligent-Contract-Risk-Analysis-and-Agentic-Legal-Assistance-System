import React from 'react';

interface ErrorMessageProps {
  message: string;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message }) => {
  return (
    <div className="error">
      <strong>Error:</strong> {message}
    </div>
  );
};

export default ErrorMessage;
