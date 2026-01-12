import { useState, useRef, useEffect } from 'react';
import { TRANSFORMERLENS_MODELS, MODEL_FAMILIES } from './transformerlens-models';

interface SearchableModelDropdownProps {
  value: string;
  onChange: (modelId: string) => void;
  disabled?: boolean;
  onClose?: () => void;
}

export const SearchableModelDropdown = ({ value, onChange, disabled, onClose }: SearchableModelDropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFamily, setSelectedFamily] = useState<string>('All');
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Focus search input when dropdown opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Filter models based on search query and family
  const filteredModels = TRANSFORMERLENS_MODELS.filter(model => {
    const matchesSearch =
      model.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.family.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesFamily = selectedFamily === 'All' || model.family === selectedFamily;

    return matchesSearch && matchesFamily;
  });

  const selectedModel = TRANSFORMERLENS_MODELS.find(m => m.id === value);
  const displayText = selectedModel ? selectedModel.name : 'More Models...';

  const handleSelectModel = (modelId: string) => {
    onChange(modelId);
    setIsOpen(false);
    setSearchQuery('');
    setSelectedFamily('All');
  };

  return (
    <div ref={dropdownRef} style={{ position: 'relative', display: 'inline-flex', gap: '0.5rem', alignItems: 'center' }}>
      {onClose && (
        <button
          onClick={onClose}
          disabled={disabled}
          style={{
            padding: '4px 12px',
            fontSize: '0.9em',
            cursor: disabled ? 'not-allowed' : 'pointer',
            backgroundColor: '#2a2a2a',
            color: '#fff',
            border: '1px solid #555',
            borderRadius: '4px',
            opacity: disabled ? 0.5 : 1
          }}
          title="Back to standard dropdown"
        >
          ← Back
        </button>
      )}
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        style={{
          padding: '4px 12px',
          fontSize: '0.9em',
          cursor: disabled ? 'not-allowed' : 'pointer',
          backgroundColor: '#2a2a2a',
          color: '#fff',
          border: '1px solid #555',
          borderRadius: '4px',
          minWidth: '200px',
          textAlign: 'left',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          opacity: disabled ? 0.5 : 1
        }}
      >
        <span>{displayText}</span>
        <span style={{ marginLeft: '8px' }}>{isOpen ? '▲' : '▼'}</span>
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          marginTop: '4px',
          backgroundColor: '#1a1a1a',
          border: '1px solid #555',
          borderRadius: '4px',
          zIndex: 1000,
          minWidth: '400px',
          maxHeight: '500px',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
        }}>
          {/* Search and Filter Header */}
          <div style={{ padding: '8px', borderBottom: '1px solid #555' }}>
            <input
              ref={inputRef}
              type="text"
              placeholder="Search models (e.g., 'Eluther', 'pythia', '160m')..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: '100%',
                padding: '6px 8px',
                fontSize: '0.9em',
                backgroundColor: '#2a2a2a',
                color: '#fff',
                border: '1px solid #555',
                borderRadius: '4px',
                marginBottom: '8px'
              }}
            />

            <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
              <button
                onClick={() => setSelectedFamily('All')}
                style={{
                  padding: '2px 8px',
                  fontSize: '0.75em',
                  backgroundColor: selectedFamily === 'All' ? '#007acc' : '#2a2a2a',
                  color: '#fff',
                  border: '1px solid #555',
                  borderRadius: '3px',
                  cursor: 'pointer'
                }}
              >
                All ({TRANSFORMERLENS_MODELS.length})
              </button>
              {MODEL_FAMILIES.map(family => {
                const count = TRANSFORMERLENS_MODELS.filter(m => m.family === family).length;
                return (
                  <button
                    key={family}
                    onClick={() => setSelectedFamily(family)}
                    style={{
                      padding: '2px 8px',
                      fontSize: '0.75em',
                      backgroundColor: selectedFamily === family ? '#007acc' : '#2a2a2a',
                      color: '#fff',
                      border: '1px solid #555',
                      borderRadius: '3px',
                      cursor: 'pointer'
                    }}
                  >
                    {family} ({count})
                  </button>
                );
              })}
            </div>
          </div>

          {/* Model List */}
          <div style={{
            overflowY: 'auto',
            maxHeight: '400px',
            padding: '4px'
          }}>
            {filteredModels.length === 0 ? (
              <div style={{ padding: '12px', textAlign: 'center', color: '#888' }}>
                No models found matching "{searchQuery}"
              </div>
            ) : (
              filteredModels.map((model) => (
                <button
                  key={model.id}
                  onClick={() => handleSelectModel(model.id)}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    textAlign: 'left',
                    backgroundColor: model.id === value ? '#007acc' : 'transparent',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '0.85em',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}
                  onMouseEnter={(e) => {
                    if (model.id !== value) {
                      e.currentTarget.style.backgroundColor = '#2a2a2a';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (model.id !== value) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }
                  }}
                >
                  <span>
                    <strong>{model.name}</strong>
                    <br />
                    <span style={{ fontSize: '0.85em', color: '#999' }}>{model.id}</span>
                  </span>
                  <span style={{
                    fontSize: '0.75em',
                    color: '#007acc',
                    backgroundColor: '#1a1a1a',
                    padding: '2px 6px',
                    borderRadius: '3px',
                    border: '1px solid #007acc'
                  }}>
                    {model.family}
                  </span>
                </button>
              ))
            )}
          </div>

          {/* Footer with count */}
          <div style={{
            padding: '6px 12px',
            borderTop: '1px solid #555',
            fontSize: '0.75em',
            color: '#888',
            textAlign: 'center'
          }}>
            Showing {filteredModels.length} of {TRANSFORMERLENS_MODELS.length} models
          </div>
        </div>
      )}
    </div>
  );
};
