from fastapi import APIRouter,Depends
from src.controllers import model_fit as controller
from src.schemas.entities import ModelFitInput,RegisterFitInput
from src.services.auth import current,editor,admin
router=APIRouter(prefix='/model-fits',tags=['model-fits'])
@router.get('')
def list_items(user=Depends(current)):return controller.list_items()
@router.post('',status_code=201)
def create(data:ModelFitInput,user=Depends(editor)):return controller.create(data)
@router.get('/{id}')
def detail(id:str,user=Depends(current)):return controller.detail(id)
@router.post('/{id}/register',status_code=201)
def register(id:str,data:RegisterFitInput,user=Depends(editor)):return controller.register(id,data)
