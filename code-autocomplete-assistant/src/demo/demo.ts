interface UserProfile {
    id: string;
    name: string;
    email: string;

    // <CURSOR HERE - suggest additional properties>
    // Expected: createdAt: Date;
    //           updatedAt: Date;
    //           isActive: boolean;
    //           roles: string[];
}

interface ApiResponse<T> {
    success: boolean;
    data: T;
    
    // <CURSOR HERE>
    // Expected: error?: string;
    //           timestamp: number;
}