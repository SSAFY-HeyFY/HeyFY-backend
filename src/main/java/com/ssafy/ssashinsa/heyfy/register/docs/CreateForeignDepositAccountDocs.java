package com.ssafy.ssashinsa.heyfy.register.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.register.dto.AccountCreationResponseDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject; // 💡 import 추가
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Tag(name = "Register", description = "계좌 등록 API")
@Operation(summary = "외화 예금 계좌 등록", description = "신한은행 API를 이용하여 새로운 외화 예금 계좌를 개설합니다.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 외화 계좌를 등록했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountCreationResponseDto.class),
                        examples = @ExampleObject( // 💡 examples 추가
                                name = "성공 응답 예시",
                                value = "{\"message\": \"정상처리 되었습니다.\", \"accountNo\": \"0019290964871122\", \"currency\": \"USD\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class)
                )
        )
})
public @interface CreateForeignDepositAccountDocs {
}