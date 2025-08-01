# Uploads Directory

This directory is used by the Flask application to temporarily store uploaded banana images for prediction.

## Purpose

- Temporary storage of user-uploaded images
- Processing location for model predictions
- Automatically managed by the Flask application

## Notes

- Files in this directory are temporary and can be cleaned periodically
- Maximum file size is limited to 16MB
- Supports common image formats (JPG, PNG)
- Directory is created automatically if it doesn't exist

## Security

- Filenames are sanitized using `secure_filename`
- File types are validated before processing
- Old files should be cleaned up regularly
