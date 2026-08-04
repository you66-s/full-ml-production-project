from core.database import session
import logging
 
logger = logging.getLogger(__name__)

def get_db():
    db = session()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database Error: {type(e).__name__} - {str(e)}")
        db.rollback()     # rollback if anything failed
        raise
    finally:
        db.close()        # always close, whether success or failure