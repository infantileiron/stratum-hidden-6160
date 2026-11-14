from src.services import model_fit as service
def list_items():return service.list_fits()
def detail(id):return service.get_fit(id)
def create(data):return service.create_fit(data)
def register(id,data):return service.register_candidate(id,data)
