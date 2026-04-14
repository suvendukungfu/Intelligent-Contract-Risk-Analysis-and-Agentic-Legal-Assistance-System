import React from 'react';

interface MilestoneSelectorProps {
  selectedMilestone: 1 | 2;
  onMilestoneChange: (milestone: 1 | 2) => void;
}

const MilestoneSelector: React.FC<MilestoneSelectorProps> = ({
  selectedMilestone,
  onMilestoneChange,
}) => {
  return (
    <div className="milestone-selector">
      <button
        className={`milestone-button ${selectedMilestone === 1 ? 'active' : ''}`}
        onClick={() => onMilestoneChange(1)}
      >
        ML Classification
      </button>
      <button
        className={`milestone-button ${selectedMilestone === 2 ? 'active' : ''}`}
        onClick={() => onMilestoneChange(2)}
      >
        Agentic Analysis
      </button>
    </div>
  );
};

export default MilestoneSelector;
