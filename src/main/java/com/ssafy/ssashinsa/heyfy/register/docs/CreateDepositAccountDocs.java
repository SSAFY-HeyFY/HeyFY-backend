package com.ssafy.ssashinsa.heyfy.register.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.register.dto.AccountCreationResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "Create Deposit Account", description = "Creates a new deposit account using the Shinhan Bank API.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "Account successfully registered.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountCreationResponseDto.class),
                        examples = @ExampleObject(
                                name = "Success Response Example",
                                value = "{\"message\": \"Processed normally.\", \"accountNo\": \"0016956302770649\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "401",
                description = "Authentication failed (e.g., missing user key).",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "Missing User Key",
                                summary = "The user key is not present in the request.",
                                value = "{\"status\": 401, \"httpError\": \"UNAUTHORIZED\", \"errorCode\": \"MISSING_USER_KEY\", \"message\": \"User key is missing.\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "404",
                description = "Not Found (user not found).",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "User Not Found",
                                summary = "User information from the JWT token could not be found.",
                                value = "{\"status\": 404, \"httpError\": \"NOT_FOUND\", \"errorCode\": \"USER_NOT_FOUND\", \"message\": \"User not found.\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "Internal Server Error (e.g., external API call failed).",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "External API Call Failed",
                                summary = "Shinhan API response error or internal server error.",
                                value = "{\"status\": 500, \"httpError\": \"INTERNAL_SERVER_ERROR\", \"errorCode\": \"API_CALL_FAILED\", \"message\": \"Shinhan API call failed.\"}"
                        )
                )
        )
})
public @interface CreateDepositAccountDocs {
}