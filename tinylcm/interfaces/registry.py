from typing import Any, Callable, Dict, List, Type, TypeVar, Generic

T = TypeVar('T')

class Registry(Generic[T]):
    def __init__(self, base_type: Type[T]):
        self._base_type = base_type
        self._registry: Dict[str, Type[T]] = {}
    
    def register(self, name: str, cls: Type[T]) -> None:
        if not issubclass(cls, self._base_type):
            raise TypeError(f"{cls.__name__} is not a subtype of {self._base_type.__name__}")
        self._registry[name] = cls
    
    def get(self, name: str) -> Type[T]:
        if name not in self._registry:
            raise KeyError(f"No component registered with name '{name}'")
        return self._registry[name]
    
    def create(self, name: str, *args, **kwargs) -> T:
        cls = self.get(name)
        return cls(*args, **kwargs)
    
    def list_registered(self) -> List[str]:
        return list(self._registry.keys())